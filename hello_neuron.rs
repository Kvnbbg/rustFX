// hello_neuron.rs - First program: A simple neuron activation demo
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

fn main() {
    let mut neuron = Neuron::new(2);
    let inputs = vec![0.5, 0.3];
    let output = neuron.activate(&inputs);
    println!("Hello Neuron! Output: {}", output);
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
        let uniform = Uniform::from(-1.0..1.0);
        let weights: Vec<f64> = (0..input_size).map(|_| uniform.sample(&mut rng)).collect();
        let bias = uniform.sample(&mut rng);
        Neuron { weights, bias, activation: 0.0 }
    }

    pub fn activate(&mut self, inputs: &[f64]) -> f64 {
        self.activation = inputs.iter().zip(self.weights.iter()).map(|(&i, &w)| i * w).sum::<f64>() + self.bias;
        self.activation = 1.0 / (1.0 + (-self.activation).exp()); // Sigmoid
        self.activation
    }
}
