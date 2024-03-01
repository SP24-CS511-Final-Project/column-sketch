use rand_distr::{Distribution, StandardNormal};

pub fn gen_dataset_normal(size: usize) -> Vec<f64> {
  let mut rng = rand::thread_rng();
  StandardNormal.sample_iter(&mut rng).take(size).collect()
}
