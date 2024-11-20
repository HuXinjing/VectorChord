use base::distance::{Distance, DistanceKind};
use base::scalar::ScalarLike;
use nalgebra::DMatrix;
use std::sync::OnceLock;

fn random_matrix(n: usize) -> DMatrix<f32> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha12Rng;
    use rand_distr::StandardNormal;
    let mut rng = ChaCha12Rng::from_seed([7; 32]);
    DMatrix::from_fn(n, n, |_, _| rng.sample(StandardNormal))
}

#[ignore]
#[test]
fn check_all_matrixs_are_full_rank() {
    let parallelism = std::thread::available_parallelism().unwrap().get();
    std::thread::scope(|scope| {
        let mut threads = vec![];
        for remainder in 0..parallelism {
            threads.push(scope.spawn(move || {
                for n in (0..=60000).filter(|x| x % parallelism == remainder) {
                    let matrix = random_matrix(n);
                    assert!(matrix.is_invertible());
                }
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }
    });
}

#[test]
fn check_matrices() {
    assert_eq!(
        orthogonal_matrix(2),
        vec![vec![-0.5424608, -0.8400813], vec![0.8400813, -0.54246056]]
    );
    assert_eq!(
        orthogonal_matrix(3),
        vec![
            vec![-0.5309615, -0.69094884, -0.49058124],
            vec![0.8222731, -0.56002235, -0.10120347],
            vec![0.20481002, 0.45712686, -0.86549866]
        ]
    );
}

fn orthogonal_matrix(n: usize) -> Vec<Vec<f32>> {
    use nalgebra::QR;
    let matrix = random_matrix(n);
    // QR decomposition is unique if the matrix is full rank
    let qr = QR::new(matrix);
    let q = qr.q();
    let mut projection = Vec::new();
    for row in q.row_iter() {
        projection.push(row.iter().copied().collect::<Vec<f32>>());
    }
    projection
}

static MATRIXS: [OnceLock<Vec<Vec<f32>>>; 1 + 60000] = [const { OnceLock::new() }; 1 + 60000];

pub fn prewarm(n: usize) {
    if n <= 60000 {
        MATRIXS[n].get_or_init(|| orthogonal_matrix(n));
    }
}

pub fn project(vector: &[f32]) -> Vec<f32> {
    use base::scalar::ScalarLike;
    let n = vector.len();
    let matrix = MATRIXS[n].get_or_init(|| orthogonal_matrix(n));
    (0..n)
        .map(|i| f32::reduce_sum_of_xy(vector, &matrix[i]))
        .collect()
}

#[derive(Debug, Clone)]
pub struct Code {
    pub dis_u_2: f32,
    pub factor_ppc: f32,
    pub factor_ip: f32,
    pub factor_err: f32,
    pub signs: Vec<u8>,
}

impl Code {
    pub fn t(&self) -> Vec<u64> {
        use quantization::utils::InfiniteByteChunks;
        let mut result = Vec::new();
        for x in InfiniteByteChunks::<_, 64>::new(self.signs.iter().copied())
            .take(self.signs.len().div_ceil(64))
        {
            let mut r = 0_u64;
            for i in 0..64 {
                r |= (x[i] as u64) << i;
            }
            result.push(r);
        }
        result
    }
}

pub fn code(dims: u32, vector: &[f32]) -> Code {
    let sum_of_abs_x = f32::reduce_sum_of_abs_x(vector);
    let sum_of_x_2 = f32::reduce_sum_of_x2(vector);
    let dis_u = sum_of_x_2.sqrt();
    let x0 = sum_of_abs_x / (sum_of_x_2 * (dims as f32)).sqrt();
    let x_x0 = dis_u / x0;
    let fac_norm = (dims as f32).sqrt();
    let max_x1 = 1.0f32 / (dims as f32 - 1.0).sqrt();
    let factor_err = 2.0f32 * max_x1 * (x_x0 * x_x0 - dis_u * dis_u).sqrt();
    let factor_ip = -2.0f32 / fac_norm * x_x0;
    let cnt_pos = vector
        .iter()
        .map(|x| x.is_sign_positive() as i32)
        .sum::<i32>();
    let cnt_neg = vector
        .iter()
        .map(|x| x.is_sign_negative() as i32)
        .sum::<i32>();
    let factor_ppc = factor_ip * (cnt_pos - cnt_neg) as f32;
    let mut signs = Vec::new();
    for i in 0..dims {
        signs.push(vector[i as usize].is_sign_positive() as u8);
    }
    Code {
        dis_u_2: sum_of_x_2,
        factor_ppc,
        factor_ip,
        factor_err,
        signs,
    }
}

pub type Lut = (f32, f32, f32, f32, (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>));

pub fn fscan_preprocess(vector: &[f32]) -> Lut {
    use quantization::quantize;
    let dis_v_2 = f32::reduce_sum_of_x2(vector);
    let (k, b, qvector) = quantize::quantize::<15>(vector);
    let qvector_sum = if vector.len() <= 4369 {
        quantize::reduce_sum_of_x_as_u16(&qvector) as f32
    } else {
        quantize::reduce_sum_of_x_as_u32(&qvector) as f32
    };
    (dis_v_2, b, k, qvector_sum, binarize(&qvector))
}

pub fn fscan_process_lowerbound(
    distance_kind: DistanceKind,
    _dims: u32,
    lut: &Lut,
    (dis_u_2, factor_ppc, factor_ip, factor_err, t): (f32, f32, f32, f32, &[u64]),
    epsilon: f32,
) -> Distance {
    match distance_kind {
        DistanceKind::L2 => {
            let &(dis_v_2, b, k, qvector_sum, ref s) = lut;
            let value = asymmetric_binary_dot_product(t, s) as u16;
            let rough = dis_u_2
                + dis_v_2
                + b * factor_ppc
                + ((2.0 * value as f32) - qvector_sum) * factor_ip * k;
            let err = factor_err * dis_v_2.sqrt();
            Distance::from_f32(rough - epsilon * err)
        }
        DistanceKind::Dot => {
            let &(dis_v_2, b, k, qvector_sum, ref s) = lut;
            let value = asymmetric_binary_dot_product(t, s) as u16;
            let rough =
                0.5 * b * factor_ppc + 0.5 * ((2.0 * value as f32) - qvector_sum) * factor_ip * k;
            let err = 0.5 * factor_err * dis_v_2.sqrt();
            Distance::from_f32(rough - epsilon * err)
        }
        DistanceKind::Hamming => unimplemented!(),
        DistanceKind::Jaccard => unimplemented!(),
    }
}

fn binarize(vector: &[u8]) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>) {
    let n = vector.len();
    let mut t0 = vec![0u64; n.div_ceil(64)];
    let mut t1 = vec![0u64; n.div_ceil(64)];
    let mut t2 = vec![0u64; n.div_ceil(64)];
    let mut t3 = vec![0u64; n.div_ceil(64)];
    for i in 0..n {
        t0[i / 64] |= (((vector[i] >> 0) & 1) as u64) << (i % 64);
        t1[i / 64] |= (((vector[i] >> 1) & 1) as u64) << (i % 64);
        t2[i / 64] |= (((vector[i] >> 2) & 1) as u64) << (i % 64);
        t3[i / 64] |= (((vector[i] >> 3) & 1) as u64) << (i % 64);
    }
    (t0, t1, t2, t3)
}

#[detect::multiversion(v2, fallback)]
fn asymmetric_binary_dot_product(x: &[u64], y: &(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>)) -> u32 {
    assert_eq!(x.len(), y.0.len());
    assert_eq!(x.len(), y.1.len());
    assert_eq!(x.len(), y.2.len());
    assert_eq!(x.len(), y.3.len());
    let n = x.len();
    let (mut t0, mut t1, mut t2, mut t3) = (0, 0, 0, 0);
    for i in 0..n {
        t0 += (x[i] & y.0[i]).count_ones();
    }
    for i in 0..n {
        t1 += (x[i] & y.1[i]).count_ones();
    }
    for i in 0..n {
        t2 += (x[i] & y.2[i]).count_ones();
    }
    for i in 0..n {
        t3 += (x[i] & y.3[i]).count_ones();
    }
    (t0 << 0) + (t1 << 1) + (t2 << 2) + (t3 << 3)
}
