use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

const WEIGHT_INIT_MIN: f64 = -1.0;
const WEIGHT_INIT_MAX: f64 = 1.0;
const SIGMOID_ONE: f64 = 1.0;

fn main() {
    let mut neuron = Neuron::new(2);
    let inputs = vec![0.5, 0.3];
    let _ = neuron.activate(&inputs);
}

#[derive(Debug, Clone)]
pub struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    activation: f64,
}

impl Neuron {
    pub fn new(input_size: usize) -> Self {
        let mut rng = thread_rng();
        let uniform = Uniform::from(WEIGHT_INIT_MIN..WEIGHT_INIT_MAX);
        let weights: Vec<f64> = (0..input_size).map(|_| uniform.sample(&mut rng)).collect();
        let bias = uniform.sample(&mut rng);
        Neuron { weights, bias, activation: 0.0 }
    }

    pub fn activate(&mut self, inputs: &[f64]) -> f64 {
        self.activation = inputs.iter().zip(self.weights.iter()).map(|(&i, &w)| i * w).sum::<f64>() + self.bias;
        self.activation = SIGMOID_ONE / (SIGMOID_ONE + (-self.activation).exp());
        self.activation
    }
}
