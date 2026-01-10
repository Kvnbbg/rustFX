// brain_fusion_net.rs - Updated with backpropagation
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

#[derive(Debug, Clone)]
pub struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    activation: f64,
    delta: f64, // For backprop
}

impl Neuron {
    pub fn new(input_size: usize) -> Self {
        let mut rng = thread_rng();
        let uniform = Uniform::from(-1.0..1.0);
        let weights: Vec<f64> = (0..input_size).map(|_| uniform.sample(&mut rng)).collect();
        let bias = uniform.sample(&mut rng);
        Neuron { weights, bias, activation: 0.0, delta: 0.0 }
    }

    pub fn activate(&mut self, inputs: &[f64]) -> f64 {
        self.activation = inputs.iter().zip(self.weights.iter()).map(|(&i, &w)| i * w).sum::<f64>() + self.bias;
        self.activation = 1.0 / (1.0 + (-self.activation).exp()); // Sigmoid
        self.activation
    }

    pub fn hebbian_update(&mut self, inputs: &[f64], learning_rate: f64) {
        for (w, &i) in self.weights.iter_mut().zip(inputs.iter()) {
            *w += learning_rate * self.activation * i;
        }
        self.bias += learning_rate * self.activation;
    }
}

#[derive(Debug, Clone)]
pub struct BrainFusionNet {
    layers: Vec<Vec<Neuron>>,
}

impl BrainFusionNet {
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 1..layer_sizes.len() {
            let layer: Vec<Neuron> = (0..layer_sizes[i]).map(|_| Neuron::new(layer_sizes[i-1])).collect();
            layers.push(layer);
        }
        BrainFusionNet { layers }
    }

    pub fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        let mut current = inputs.to_vec();
        for layer in self.layers.iter_mut() {
            let mut next = vec![0.0; layer.len()];
            for (i, neuron) in layer.iter_mut().enumerate() {
                next[i] = neuron.activate(&current);
            }
            current = next;
        }
        current
    }

    pub fn backpropagate(&mut self, inputs: &[f64], targets: &[f64], learning_rate: f64) {
        let outputs = self.forward(inputs);

        // Output layer deltas
        let output_layer = self.layers.len() - 1;
        for (i, neuron) in self.layers[output_layer].iter_mut().enumerate() {
            let error = targets[i] - outputs[i];
            neuron.delta = error * outputs[i] * (1.0 - outputs[i]); // Sigmoid derivative
        }

        // Hidden layers deltas
        for l in (0..output_layer).rev() {
            for (i, neuron) in self.layers[l].iter_mut().enumerate() {
                let error: f64 = self.layers[l+1].iter().map(|n| n.delta * n.weights[i]).sum();
                neuron.delta = error * neuron.activation * (1.0 - neuron.activation);
            }
        }

        // Update weights
        let mut current = inputs.to_vec();
        for (l, layer) in self.layers.iter_mut().enumerate() {
            let prev = if l == 0 { inputs } else { &self.layers[l-1].iter().map(|n| n.activation).collect::<Vec<_>>() };
            for neuron in layer.iter_mut() {
                for (w, &input) in neuron.weights.iter_mut().zip(prev.iter()) {
                    *w += learning_rate * neuron.delta * input;
                }
                neuron.bias += learning_rate * neuron.delta;
            }
            current = layer.iter().map(|n| n.activation).collect();
        }
    }

    pub fn train_hebbian(&mut self, inputs: &[f64], learning_rate: f64) {
        self.forward(inputs);
        for layer in self.layers.iter_mut() {
            let prev = /* get previous activations */;
            for neuron in layer.iter_mut() {
                neuron.hebbian_update(&prev, learning_rate);
            }
        }
    }
}
