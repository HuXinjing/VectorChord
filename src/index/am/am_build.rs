use crate::datatype::typmod::Typmod;
use crate::index::am::{Reloption, ctid_to_pointer};
use crate::index::opclass::{Opfamily, opfamily};
use crate::index::projection::RandomProject;
use crate::index::storage::PostgresRelation;
use algorithm::operator::{Dot, L2, Op, Vector};
use algorithm::types::*;
use half::f16;
use pgrx::pg_sys::Datum;
use rand::Rng;
use simd::Floating;
use std::num::NonZeroU64;
use std::sync::Arc;
use vector::vect::VectOwned;
use vector::{VectorBorrowed, VectorOwned};

#[derive(Debug, Clone)]
struct Heap {
    heap_relation: pgrx::pg_sys::Relation,
    index_relation: pgrx::pg_sys::Relation,
    index_info: *mut pgrx::pg_sys::IndexInfo,
    opfamily: Opfamily,
    scan: *mut pgrx::pg_sys::TableScanDescData,
}

impl Heap {
    fn traverse<V: Vector, F: FnMut((NonZeroU64, V))>(&self, progress: bool, callback: F) {
        pub struct State<'a, F> {
            pub this: &'a Heap,
            pub callback: F,
        }
        #[pgrx::pg_guard]
        unsafe extern "C" fn call<F, V: Vector>(
            _index_relation: pgrx::pg_sys::Relation,
            ctid: pgrx::pg_sys::ItemPointer,
            values: *mut Datum,
            is_null: *mut bool,
            _tuple_is_alive: bool,
            state: *mut core::ffi::c_void,
        ) where
            F: FnMut((NonZeroU64, V)),
        {
            let state = unsafe { &mut *state.cast::<State<F>>() };
            let opfamily = state.this.opfamily;
            let vector = unsafe { opfamily.input_vector(*values.add(0), *is_null.add(0)) };
            let pointer = unsafe { ctid_to_pointer(ctid.read()) };
            if let Some(vector) = vector {
                (state.callback)((pointer, V::from_owned(vector)));
            }
        }
        let table_am = unsafe { &*(*self.heap_relation).rd_tableam };
        let mut state = State {
            this: self,
            callback,
        };
        unsafe {
            table_am.index_build_range_scan.unwrap()(
                self.heap_relation,
                self.index_relation,
                self.index_info,
                true,
                false,
                progress,
                0,
                pgrx::pg_sys::InvalidBlockNumber,
                Some(call::<F, V>),
                (&mut state) as *mut State<F> as *mut _,
                self.scan,
            );
        }
    }
}

#[derive(Debug, Clone)]
struct PostgresReporter {}

impl PostgresReporter {
    fn tuples_total(&mut self, tuples_total: u64) {
        unsafe {
            pgrx::pg_sys::pgstat_progress_update_param(
                pgrx::pg_sys::PROGRESS_CREATEIDX_TUPLES_TOTAL as _,
                tuples_total as _,
            );
        }
    }
    fn tuples_done(&mut self, tuples_done: u64) {
        unsafe {
            pgrx::pg_sys::pgstat_progress_update_param(
                pgrx::pg_sys::PROGRESS_CREATEIDX_TUPLES_DONE as _,
                tuples_done as _,
            );
        }
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn ambuild(
    heap_relation: pgrx::pg_sys::Relation,
    index_relation: pgrx::pg_sys::Relation,
    index_info: *mut pgrx::pg_sys::IndexInfo,
) -> *mut pgrx::pg_sys::IndexBuildResult {
    use validator::Validate;
    let (vector_options, vchordrq_options) = unsafe { options(index_relation) };
    if let Err(errors) = Validate::validate(&vector_options) {
        pgrx::error!("error while validating options: {}", errors);
    }
    if vector_options.dims == 0 {
        pgrx::error!("error while validating options: dimension cannot be 0");
    }
    if vector_options.dims > 60000 {
        pgrx::error!("error while validating options: dimension is too large");
    }
    if let Err(errors) = Validate::validate(&vchordrq_options) {
        pgrx::error!("error while validating options: {}", errors);
    }
    let opfamily = unsafe { opfamily(index_relation) };
    let heap = Heap {
        heap_relation,
        index_relation,
        index_info,
        opfamily,
        scan: std::ptr::null_mut(),
    };
    let index = unsafe { PostgresRelation::new(index_relation) };
    let mut reporter = PostgresReporter {};
    let structures = match vchordrq_options.build.clone() {
        VchordrqBuildOptions::External(external_build) => {
            make_external_build(vector_options.clone(), opfamily, external_build.clone())
        }
        VchordrqBuildOptions::Internal(internal_build) => {
            let mut tuples_total = 0_u64;
            let samples = {
                let mut rand = rand::thread_rng();
                let max_number_of_samples = internal_build
                    .lists
                    .last()
                    .unwrap()
                    .saturating_mul(internal_build.sampling_factor);
                let mut samples = Vec::new();
                let mut number_of_samples = 0_u32;
                match opfamily.vector_kind() {
                    VectorKind::Vecf32 => {
                        heap.traverse(false, |(_, vector): (_, VectOwned<f32>)| {
                            let vector = vector.as_borrowed();
                            assert_eq!(
                                vector_options.dims,
                                vector.dims(),
                                "invalid vector dimensions"
                            );
                            if number_of_samples < max_number_of_samples {
                                samples.push(VectOwned::<f32>::build_to_vecf32(vector));
                                number_of_samples += 1;
                            } else {
                                let index = rand.gen_range(0..max_number_of_samples) as usize;
                                samples[index] = VectOwned::<f32>::build_to_vecf32(vector);
                            }
                            tuples_total += 1;
                        });
                    }
                    VectorKind::Vecf16 => {
                        heap.traverse(false, |(_, vector): (_, VectOwned<f16>)| {
                            let vector = vector.as_borrowed();
                            assert_eq!(
                                vector_options.dims,
                                vector.dims(),
                                "invalid vector dimensions"
                            );
                            if number_of_samples < max_number_of_samples {
                                samples.push(VectOwned::<f16>::build_to_vecf32(vector));
                                number_of_samples += 1;
                            } else {
                                let index = rand.gen_range(0..max_number_of_samples) as usize;
                                samples[index] = VectOwned::<f16>::build_to_vecf32(vector);
                            }
                            tuples_total += 1;
                        });
                    }
                }
                samples
            };
            reporter.tuples_total(tuples_total);
            make_internal_build(vector_options.clone(), internal_build.clone(), samples)
        }
    };
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2) => algorithm::build::<Op<VectOwned<f32>, L2>>(
            vector_options,
            vchordrq_options,
            index.clone(),
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
        (VectorKind::Vecf32, DistanceKind::Dot) => algorithm::build::<Op<VectOwned<f32>, Dot>>(
            vector_options,
            vchordrq_options,
            index.clone(),
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
        (VectorKind::Vecf16, DistanceKind::L2) => algorithm::build::<Op<VectOwned<f16>, L2>>(
            vector_options,
            vchordrq_options,
            index.clone(),
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
        (VectorKind::Vecf16, DistanceKind::Dot) => algorithm::build::<Op<VectOwned<f16>, Dot>>(
            vector_options,
            vchordrq_options,
            index.clone(),
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
    }
    if let Some(leader) =
        unsafe { VchordrqLeader::enter(heap_relation, index_relation, (*index_info).ii_Concurrent) }
    {
        unsafe {
            parallel_build(
                index_relation,
                heap_relation,
                index_info,
                leader.tablescandesc,
                leader.vchordrqshared,
                Some(reporter),
            );
            leader.wait();
            let nparticipants = leader.nparticipants;
            loop {
                pgrx::pg_sys::SpinLockAcquire(&raw mut (*leader.vchordrqshared).mutex);
                if (*leader.vchordrqshared).nparticipantsdone == nparticipants {
                    pgrx::pg_sys::SpinLockRelease(&raw mut (*leader.vchordrqshared).mutex);
                    break;
                }
                pgrx::pg_sys::SpinLockRelease(&raw mut (*leader.vchordrqshared).mutex);
                pgrx::pg_sys::ConditionVariableSleep(
                    &raw mut (*leader.vchordrqshared).workersdonecv,
                    pgrx::pg_sys::WaitEventIPC::WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN,
                );
            }
            pgrx::pg_sys::ConditionVariableCancelSleep();
        }
    } else {
        let mut indtuples = 0;
        reporter.tuples_done(indtuples);
        let relation = unsafe { PostgresRelation::new(index_relation) };
        match (opfamily.vector_kind(), opfamily.distance_kind()) {
            (VectorKind::Vecf32, DistanceKind::L2) => {
                heap.traverse(true, |(pointer, vector): (_, VectOwned<f32>)| {
                    algorithm::insert::<Op<VectOwned<f32>, L2>>(
                        relation.clone(),
                        pointer,
                        RandomProject::project(vector.as_borrowed()),
                    );
                    indtuples += 1;
                    reporter.tuples_done(indtuples);
                });
            }
            (VectorKind::Vecf32, DistanceKind::Dot) => {
                heap.traverse(true, |(pointer, vector): (_, VectOwned<f32>)| {
                    algorithm::insert::<Op<VectOwned<f32>, Dot>>(
                        relation.clone(),
                        pointer,
                        RandomProject::project(vector.as_borrowed()),
                    );
                    indtuples += 1;
                    reporter.tuples_done(indtuples);
                });
            }
            (VectorKind::Vecf16, DistanceKind::L2) => {
                heap.traverse(true, |(pointer, vector): (_, VectOwned<f16>)| {
                    algorithm::insert::<Op<VectOwned<f16>, L2>>(
                        relation.clone(),
                        pointer,
                        RandomProject::project(vector.as_borrowed()),
                    );
                    indtuples += 1;
                    reporter.tuples_done(indtuples);
                });
            }
            (VectorKind::Vecf16, DistanceKind::Dot) => {
                heap.traverse(true, |(pointer, vector): (_, VectOwned<f16>)| {
                    algorithm::insert::<Op<VectOwned<f16>, Dot>>(
                        relation.clone(),
                        pointer,
                        RandomProject::project(vector.as_borrowed()),
                    );
                    indtuples += 1;
                    reporter.tuples_done(indtuples);
                });
            }
        }
    }
    let check = || {
        pgrx::check_for_interrupts!();
    };
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2) => {
            algorithm::maintain::<Op<VectOwned<f32>, L2>>(index, check);
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            algorithm::maintain::<Op<VectOwned<f32>, Dot>>(index, check);
        }
        (VectorKind::Vecf16, DistanceKind::L2) => {
            algorithm::maintain::<Op<VectOwned<f16>, L2>>(index, check);
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            algorithm::maintain::<Op<VectOwned<f16>, Dot>>(index, check);
        }
    }
    unsafe { pgrx::pgbox::PgBox::<pgrx::pg_sys::IndexBuildResult>::alloc0().into_pg() }
}

struct VchordrqShared {
    /* Immutable state */
    heaprelid: pgrx::pg_sys::Oid,
    indexrelid: pgrx::pg_sys::Oid,
    isconcurrent: bool,

    /* Worker progress */
    workersdonecv: pgrx::pg_sys::ConditionVariable,

    /* Mutex for mutable state */
    mutex: pgrx::pg_sys::slock_t,

    /* Mutable state */
    nparticipantsdone: i32,
    indtuples: u64,
}

fn is_mvcc_snapshot(snapshot: *mut pgrx::pg_sys::SnapshotData) -> bool {
    matches!(
        unsafe { (*snapshot).snapshot_type },
        pgrx::pg_sys::SnapshotType::SNAPSHOT_MVCC
            | pgrx::pg_sys::SnapshotType::SNAPSHOT_HISTORIC_MVCC
    )
}

struct VchordrqLeader {
    pcxt: *mut pgrx::pg_sys::ParallelContext,
    nparticipants: i32,
    vchordrqshared: *mut VchordrqShared,
    tablescandesc: *mut pgrx::pg_sys::ParallelTableScanDescData,
    snapshot: pgrx::pg_sys::Snapshot,
}

impl VchordrqLeader {
    pub unsafe fn enter(
        heap_relation: pgrx::pg_sys::Relation,
        index_relation: pgrx::pg_sys::Relation,
        isconcurrent: bool,
    ) -> Option<Self> {
        unsafe fn compute_parallel_workers(
            heap_relation: pgrx::pg_sys::Relation,
            index_relation: pgrx::pg_sys::Relation,
        ) -> i32 {
            unsafe {
                if pgrx::pg_sys::plan_create_index_workers(
                    (*heap_relation).rd_id,
                    (*index_relation).rd_id,
                ) == 0
                {
                    return 0;
                }
                if !(*heap_relation).rd_options.is_null() {
                    let std_options = (*heap_relation)
                        .rd_options
                        .cast::<pgrx::pg_sys::StdRdOptions>();
                    std::cmp::min(
                        (*std_options).parallel_workers,
                        pgrx::pg_sys::max_parallel_maintenance_workers,
                    )
                } else {
                    pgrx::pg_sys::max_parallel_maintenance_workers
                }
            }
        }

        let request = unsafe { compute_parallel_workers(heap_relation, index_relation) };
        if request <= 0 {
            return None;
        }

        unsafe {
            pgrx::pg_sys::EnterParallelMode();
        }
        let pcxt = unsafe {
            pgrx::pg_sys::CreateParallelContext(
                c"vchord".as_ptr(),
                c"vchordrq_parallel_build_main".as_ptr(),
                request,
            )
        };

        let snapshot = if isconcurrent {
            unsafe { pgrx::pg_sys::RegisterSnapshot(pgrx::pg_sys::GetTransactionSnapshot()) }
        } else {
            &raw mut pgrx::pg_sys::SnapshotAnyData
        };

        fn estimate_chunk(e: &mut pgrx::pg_sys::shm_toc_estimator, x: usize) {
            e.space_for_chunks += x.next_multiple_of(pgrx::pg_sys::ALIGNOF_BUFFER as _);
        }
        fn estimate_keys(e: &mut pgrx::pg_sys::shm_toc_estimator, x: usize) {
            e.number_of_keys += x;
        }
        let est_tablescandesc =
            unsafe { pgrx::pg_sys::table_parallelscan_estimate(heap_relation, snapshot) };
        unsafe {
            estimate_chunk(&mut (*pcxt).estimator, size_of::<VchordrqShared>());
            estimate_keys(&mut (*pcxt).estimator, 1);
            estimate_chunk(&mut (*pcxt).estimator, est_tablescandesc);
            estimate_keys(&mut (*pcxt).estimator, 1);
        }

        unsafe {
            pgrx::pg_sys::InitializeParallelDSM(pcxt);
            if (*pcxt).seg.is_null() {
                if is_mvcc_snapshot(snapshot) {
                    pgrx::pg_sys::UnregisterSnapshot(snapshot);
                }
                pgrx::pg_sys::DestroyParallelContext(pcxt);
                pgrx::pg_sys::ExitParallelMode();
                return None;
            }
        }

        let vchordrqshared = unsafe {
            let vchordrqshared =
                pgrx::pg_sys::shm_toc_allocate((*pcxt).toc, size_of::<VchordrqShared>())
                    .cast::<VchordrqShared>();
            vchordrqshared.write(VchordrqShared {
                heaprelid: (*heap_relation).rd_id,
                indexrelid: (*index_relation).rd_id,
                isconcurrent,
                workersdonecv: std::mem::zeroed(),
                mutex: std::mem::zeroed(),
                nparticipantsdone: 0,
                indtuples: 0,
            });
            pgrx::pg_sys::ConditionVariableInit(&raw mut (*vchordrqshared).workersdonecv);
            pgrx::pg_sys::SpinLockInit(&raw mut (*vchordrqshared).mutex);
            vchordrqshared
        };

        let tablescandesc = unsafe {
            let tablescandesc = pgrx::pg_sys::shm_toc_allocate((*pcxt).toc, est_tablescandesc)
                .cast::<pgrx::pg_sys::ParallelTableScanDescData>();
            pgrx::pg_sys::table_parallelscan_initialize(heap_relation, tablescandesc, snapshot);
            tablescandesc
        };

        unsafe {
            pgrx::pg_sys::shm_toc_insert((*pcxt).toc, 0xA000000000000001, vchordrqshared.cast());
            pgrx::pg_sys::shm_toc_insert((*pcxt).toc, 0xA000000000000002, tablescandesc.cast());
        }

        unsafe {
            pgrx::pg_sys::LaunchParallelWorkers(pcxt);
        }

        let nworkers_launched = unsafe { (*pcxt).nworkers_launched };

        unsafe {
            if nworkers_launched == 0 {
                pgrx::pg_sys::WaitForParallelWorkersToFinish(pcxt);
                if is_mvcc_snapshot(snapshot) {
                    pgrx::pg_sys::UnregisterSnapshot(snapshot);
                }
                pgrx::pg_sys::DestroyParallelContext(pcxt);
                pgrx::pg_sys::ExitParallelMode();
                return None;
            }
        }

        Some(Self {
            pcxt,
            nparticipants: nworkers_launched + 1,
            vchordrqshared,
            tablescandesc,
            snapshot,
        })
    }

    pub fn wait(&self) {
        unsafe {
            pgrx::pg_sys::WaitForParallelWorkersToAttach(self.pcxt);
        }
    }
}

impl Drop for VchordrqLeader {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            unsafe {
                pgrx::pg_sys::WaitForParallelWorkersToFinish(self.pcxt);
                if is_mvcc_snapshot(self.snapshot) {
                    pgrx::pg_sys::UnregisterSnapshot(self.snapshot);
                }
                pgrx::pg_sys::DestroyParallelContext(self.pcxt);
                pgrx::pg_sys::ExitParallelMode();
            }
        }
    }
}

#[pgrx::pg_guard]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vchordrq_parallel_build_main(
    _seg: *mut pgrx::pg_sys::dsm_segment,
    toc: *mut pgrx::pg_sys::shm_toc,
) {
    let vchordrqshared = unsafe {
        pgrx::pg_sys::shm_toc_lookup(toc, 0xA000000000000001, false).cast::<VchordrqShared>()
    };
    let tablescandesc = unsafe {
        pgrx::pg_sys::shm_toc_lookup(toc, 0xA000000000000002, false)
            .cast::<pgrx::pg_sys::ParallelTableScanDescData>()
    };
    let heap_lockmode;
    let index_lockmode;
    if unsafe { !(*vchordrqshared).isconcurrent } {
        heap_lockmode = pgrx::pg_sys::ShareLock as pgrx::pg_sys::LOCKMODE;
        index_lockmode = pgrx::pg_sys::AccessExclusiveLock as pgrx::pg_sys::LOCKMODE;
    } else {
        heap_lockmode = pgrx::pg_sys::ShareUpdateExclusiveLock as pgrx::pg_sys::LOCKMODE;
        index_lockmode = pgrx::pg_sys::RowExclusiveLock as pgrx::pg_sys::LOCKMODE;
    }
    let heap = unsafe { pgrx::pg_sys::table_open((*vchordrqshared).heaprelid, heap_lockmode) };
    let index = unsafe { pgrx::pg_sys::index_open((*vchordrqshared).indexrelid, index_lockmode) };
    let index_info = unsafe { pgrx::pg_sys::BuildIndexInfo(index) };
    unsafe {
        (*index_info).ii_Concurrent = (*vchordrqshared).isconcurrent;
    }

    unsafe {
        parallel_build(index, heap, index_info, tablescandesc, vchordrqshared, None);
    }

    unsafe {
        pgrx::pg_sys::index_close(index, index_lockmode);
        pgrx::pg_sys::table_close(heap, heap_lockmode);
    }
}

unsafe fn parallel_build(
    index_relation: pgrx::pg_sys::Relation,
    heap_relation: pgrx::pg_sys::Relation,
    index_info: *mut pgrx::pg_sys::IndexInfo,
    tablescandesc: *mut pgrx::pg_sys::ParallelTableScanDescData,
    vchordrqshared: *mut VchordrqShared,
    mut reporter: Option<PostgresReporter>,
) {
    let index = unsafe { PostgresRelation::new(index_relation) };

    let scan = unsafe { pgrx::pg_sys::table_beginscan_parallel(heap_relation, tablescandesc) };
    let opfamily = unsafe { opfamily(index_relation) };
    let heap = Heap {
        heap_relation,
        index_relation,
        index_info,
        opfamily,
        scan,
    };
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2) => {
            heap.traverse(true, |(pointer, vector): (_, VectOwned<f32>)| {
                algorithm::insert::<Op<VectOwned<f32>, L2>>(
                    index.clone(),
                    pointer,
                    RandomProject::project(vector.as_borrowed()),
                );
                unsafe {
                    let indtuples;
                    {
                        pgrx::pg_sys::SpinLockAcquire(&raw mut (*vchordrqshared).mutex);
                        (*vchordrqshared).indtuples += 1;
                        indtuples = (*vchordrqshared).indtuples;
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*vchordrqshared).mutex);
                    }
                    if let Some(reporter) = reporter.as_mut() {
                        reporter.tuples_done(indtuples);
                    }
                }
            });
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            heap.traverse(true, |(pointer, vector): (_, VectOwned<f32>)| {
                algorithm::insert::<Op<VectOwned<f32>, Dot>>(
                    index.clone(),
                    pointer,
                    RandomProject::project(vector.as_borrowed()),
                );
                unsafe {
                    let indtuples;
                    {
                        pgrx::pg_sys::SpinLockAcquire(&raw mut (*vchordrqshared).mutex);
                        (*vchordrqshared).indtuples += 1;
                        indtuples = (*vchordrqshared).indtuples;
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*vchordrqshared).mutex);
                    }
                    if let Some(reporter) = reporter.as_mut() {
                        reporter.tuples_done(indtuples);
                    }
                }
            });
        }
        (VectorKind::Vecf16, DistanceKind::L2) => {
            heap.traverse(true, |(pointer, vector): (_, VectOwned<f16>)| {
                algorithm::insert::<Op<VectOwned<f16>, L2>>(
                    index.clone(),
                    pointer,
                    RandomProject::project(vector.as_borrowed()),
                );
                unsafe {
                    let indtuples;
                    {
                        pgrx::pg_sys::SpinLockAcquire(&raw mut (*vchordrqshared).mutex);
                        (*vchordrqshared).indtuples += 1;
                        indtuples = (*vchordrqshared).indtuples;
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*vchordrqshared).mutex);
                    }
                    if let Some(reporter) = reporter.as_mut() {
                        reporter.tuples_done(indtuples);
                    }
                }
            });
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            heap.traverse(true, |(pointer, vector): (_, VectOwned<f16>)| {
                algorithm::insert::<Op<VectOwned<f16>, Dot>>(
                    index.clone(),
                    pointer,
                    RandomProject::project(vector.as_borrowed()),
                );
                unsafe {
                    let indtuples;
                    {
                        pgrx::pg_sys::SpinLockAcquire(&raw mut (*vchordrqshared).mutex);
                        (*vchordrqshared).indtuples += 1;
                        indtuples = (*vchordrqshared).indtuples;
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*vchordrqshared).mutex);
                    }
                    if let Some(reporter) = reporter.as_mut() {
                        reporter.tuples_done(indtuples);
                    }
                }
            });
        }
    }
    unsafe {
        pgrx::pg_sys::SpinLockAcquire(&raw mut (*vchordrqshared).mutex);
        (*vchordrqshared).nparticipantsdone += 1;
        pgrx::pg_sys::SpinLockRelease(&raw mut (*vchordrqshared).mutex);
        pgrx::pg_sys::ConditionVariableSignal(&raw mut (*vchordrqshared).workersdonecv);
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn ambuildempty(_index_relation: pgrx::pg_sys::Relation) {
    pgrx::error!("Unlogged indexes are not supported.");
}

unsafe fn options(
    index_relation: pgrx::pg_sys::Relation,
) -> (VectorOptions, VchordrqIndexingOptions) {
    let att = unsafe { &mut *(*index_relation).rd_att };
    let atts = unsafe { att.attrs.as_slice(att.natts as _) };
    if atts.is_empty() {
        pgrx::error!("indexing on no columns is not supported");
    }
    if atts.len() != 1 {
        pgrx::error!("multicolumn index is not supported");
    }
    // get dims
    let typmod = Typmod::parse_from_i32(atts[0].type_mod()).unwrap();
    let dims = if let Some(dims) = typmod.dims() {
        dims.get()
    } else {
        pgrx::error!(
            "Dimensions type modifier of a vector column is needed for building the index."
        );
    };
    // get v, d
    let opfamily = unsafe { opfamily(index_relation) };
    let vector = VectorOptions {
        dims,
        v: opfamily.vector_kind(),
        d: opfamily.distance_kind(),
    };
    // get indexing, segment, optimizing
    let rabitq = 'rabitq: {
        let reloption = unsafe { (*index_relation).rd_options as *const Reloption };
        if reloption.is_null() || unsafe { (*reloption).options == 0 } {
            break 'rabitq Default::default();
        }
        let s = unsafe { Reloption::options(reloption) }.to_string_lossy();
        match toml::from_str::<VchordrqIndexingOptions>(&s) {
            Ok(p) => p,
            Err(e) => pgrx::error!("failed to parse options: {}", e),
        }
    };
    (vector, rabitq)
}

pub fn make_internal_build(
    vector_options: VectorOptions,
    internal_build: VchordrqInternalBuildOptions,
    mut samples: Vec<Vec<f32>>,
) -> Vec<Structure<Vec<f32>>> {
    use std::iter::once;
    for sample in samples.iter_mut() {
        *sample = crate::index::projection::project(sample);
    }
    let mut result = Vec::<Structure<Vec<f32>>>::new();
    for w in internal_build.lists.iter().rev().copied().chain(once(1)) {
        let means = k_means::RayonParallelism::scoped(
            internal_build.build_threads as _,
            Arc::new(|| {
                pgrx::check_for_interrupts!();
            }),
            |parallelism| {
                k_means::k_means(
                    parallelism,
                    w as usize,
                    vector_options.dims as usize,
                    if let Some(structure) = result.last() {
                        &structure.means
                    } else {
                        &samples
                    },
                    internal_build.spherical_centroids,
                    10,
                )
            },
        )
        .expect("failed to create thread pool");
        if let Some(structure) = result.last() {
            let mut children = vec![Vec::new(); means.len()];
            for i in 0..structure.len() as u32 {
                let target = k_means::k_means_lookup(&structure.means[i as usize], &means);
                children[target].push(i);
            }
            let (means, children) = std::iter::zip(means, children)
                .filter(|(_, x)| !x.is_empty())
                .unzip::<_, _, Vec<_>, Vec<_>>();
            result.push(Structure { means, children });
        } else {
            let children = vec![Vec::new(); means.len()];
            result.push(Structure { means, children });
        }
    }
    result
}

pub fn make_external_build(
    vector_options: VectorOptions,
    _opfamily: Opfamily,
    external_build: VchordrqExternalBuildOptions,
) -> Vec<Structure<Vec<f32>>> {
    use std::collections::BTreeMap;
    let VchordrqExternalBuildOptions { table } = external_build;
    let mut parents = BTreeMap::new();
    let mut vectors = BTreeMap::new();
    pgrx::spi::Spi::connect(|client| {
        use crate::datatype::memory_vector::VectorOutput;
        use pgrx::pg_sys::panic::ErrorReportable;
        use vector::VectorBorrowed;
        let schema_query = "SELECT n.nspname::TEXT 
            FROM pg_catalog.pg_extension e
            LEFT JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
            WHERE e.extname = 'vector';";
        let pgvector_schema: String = client
            .select(schema_query, None, None)
            .unwrap_or_report()
            .first()
            .get_by_name("nspname")
            .expect("external build: cannot get schema of pgvector")
            .expect("external build: cannot get schema of pgvector");
        let dump_query =
            format!("SELECT id, parent, vector::{pgvector_schema}.vector FROM {table};");
        let centroids = client.select(&dump_query, None, None).unwrap_or_report();
        for row in centroids {
            let id: Option<i32> = row.get_by_name("id").unwrap();
            let parent: Option<i32> = row.get_by_name("parent").unwrap();
            let vector: Option<VectorOutput> = row.get_by_name("vector").unwrap();
            let id = id.expect("external build: id could not be NULL");
            let vector = vector.expect("external build: vector could not be NULL");
            let pop = parents.insert(id, parent);
            if pop.is_some() {
                pgrx::error!(
                    "external build: there are at least two lines have same id, id = {id}"
                );
            }
            if vector_options.dims != vector.as_borrowed().dims() {
                pgrx::error!("external build: incorrect dimension, id = {id}");
            }
            vectors.insert(
                id,
                crate::index::projection::project(vector.as_borrowed().slice()),
            );
        }
    });
    if parents.len() >= 2 && parents.values().all(|x| x.is_none()) {
        // if there are more than one vertexs and no edges,
        // assume there is an implicit root
        let n = parents.len();
        let mut result = Vec::new();
        result.push(Structure {
            means: vectors.values().cloned().collect::<Vec<_>>(),
            children: vec![Vec::new(); n],
        });
        result.push(Structure {
            means: vec![{
                // compute the vector on root, without normalizing it
                let mut sum = vec![0.0f32; vector_options.dims as _];
                for vector in vectors.values() {
                    f32::vector_add_inplace(&mut sum, vector);
                }
                f32::vector_mul_scalar_inplace(&mut sum, 1.0 / n as f32);
                sum
            }],
            children: vec![(0..n as u32).collect()],
        });
        return result;
    }
    let mut children = parents
        .keys()
        .map(|x| (*x, Vec::new()))
        .collect::<BTreeMap<_, _>>();
    let mut root = None;
    for (&id, &parent) in parents.iter() {
        if let Some(parent) = parent {
            if let Some(parent) = children.get_mut(&parent) {
                parent.push(id);
            } else {
                pgrx::error!("external build: parent does not exist, id = {id}, parent = {parent}");
            }
        } else {
            if let Some(root) = root {
                pgrx::error!("external build: two root, id = {root}, id = {id}");
            } else {
                root = Some(id);
            }
        }
    }
    let Some(root) = root else {
        pgrx::error!("external build: there are no root");
    };
    let mut heights = BTreeMap::<_, _>::new();
    fn dfs_for_heights(
        heights: &mut BTreeMap<i32, Option<u32>>,
        children: &BTreeMap<i32, Vec<i32>>,
        u: i32,
    ) {
        if heights.contains_key(&u) {
            pgrx::error!("external build: detect a cycle, id = {u}");
        }
        heights.insert(u, None);
        let mut height = None;
        for &v in children[&u].iter() {
            dfs_for_heights(heights, children, v);
            let new = heights[&v].unwrap() + 1;
            if let Some(height) = height {
                if height != new {
                    pgrx::error!("external build: two heights, id = {u}");
                }
            } else {
                height = Some(new);
            }
        }
        if height.is_none() {
            height = Some(1);
        }
        heights.insert(u, height);
    }
    dfs_for_heights(&mut heights, &children, root);
    let heights = heights
        .into_iter()
        .map(|(k, v)| (k, v.expect("not a connected graph")))
        .collect::<BTreeMap<_, _>>();
    if !(1..=8).contains(&(heights[&root] - 1)) {
        pgrx::error!(
            "external build: unexpected tree height, height = {}",
            heights[&root]
        );
    }
    let mut cursors = vec![0_u32; 1 + heights[&root] as usize];
    let mut labels = BTreeMap::new();
    for id in parents.keys().copied() {
        let height = heights[&id];
        let cursor = cursors[height as usize];
        labels.insert(id, (height, cursor));
        cursors[height as usize] += 1;
    }
    fn extract(
        height: u32,
        labels: &BTreeMap<i32, (u32, u32)>,
        vectors: &BTreeMap<i32, Vec<f32>>,
        children: &BTreeMap<i32, Vec<i32>>,
    ) -> (Vec<Vec<f32>>, Vec<Vec<u32>>) {
        labels
            .iter()
            .filter(|(_, (h, _))| *h == height)
            .map(|(id, _)| {
                (
                    vectors[id].clone(),
                    children[id].iter().map(|id| labels[id].1).collect(),
                )
            })
            .unzip()
    }
    let mut result = Vec::new();
    for height in 1..=heights[&root] {
        let (means, children) = extract(height, &labels, &vectors, &children);
        result.push(Structure { means, children });
    }
    result
}

pub fn map_structures<T, U>(x: Vec<Structure<T>>, f: impl Fn(T) -> U + Copy) -> Vec<Structure<U>> {
    x.into_iter()
        .map(|Structure { means, children }| Structure {
            means: means.into_iter().map(f).collect(),
            children,
        })
        .collect()
}

pub trait InternalBuild: VectorOwned {
    fn build_to_vecf32(vector: Self::Borrowed<'_>) -> Vec<f32>;

    fn build_from_vecf32(x: &[f32]) -> Self;
}

impl InternalBuild for VectOwned<f32> {
    fn build_to_vecf32(vector: Self::Borrowed<'_>) -> Vec<f32> {
        vector.slice().to_vec()
    }

    fn build_from_vecf32(x: &[f32]) -> Self {
        Self::new(x.to_vec())
    }
}

impl InternalBuild for VectOwned<f16> {
    fn build_to_vecf32(vector: Self::Borrowed<'_>) -> Vec<f32> {
        f16::vector_to_f32(vector.slice())
    }

    fn build_from_vecf32(x: &[f32]) -> Self {
        Self::new(f16::vector_from_f32(x))
    }
}
