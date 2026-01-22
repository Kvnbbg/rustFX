use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

const WEIGHT_INIT_MIN: f64 = -1.0;
const WEIGHT_INIT_MAX: f64 = 1.0;
const SIGMOID_ONE: f64 = 1.0;

#[derive(Debug, Clone)]
pub struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    activation: f64,
    delta: f64,
}

impl Neuron {
    /// Creates a neuron with randomized weights sized to the input vector.
    pub fn new(input_size: usize) -> Self {
        let mut rng = thread_rng();
        let uniform = Uniform::from(WEIGHT_INIT_MIN..WEIGHT_INIT_MAX);
        let weights: Vec<f64> = (0..input_size).map(|_| uniform.sample(&mut rng)).collect();
        let bias = uniform.sample(&mut rng);
        Neuron { weights, bias, activation: 0.0, delta: 0.0 }
    }

    /// Computes the sigmoid activation for the provided inputs.
    pub fn activate(&mut self, inputs: &[f64]) -> f64 {
        self.activation = inputs.iter().zip(self.weights.iter()).map(|(&i, &w)| i * w).sum::<f64>() + self.bias;
        self.activation = SIGMOID_ONE / (SIGMOID_ONE + (-self.activation).exp());
        self.activation
    }

    /// Applies a Hebbian update using the neuron activation and inputs.
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
    /// Builds a multi-layer network given the number of neurons per layer.
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 1..layer_sizes.len() {
            let layer: Vec<Neuron> = (0..layer_sizes[i]).map(|_| Neuron::new(layer_sizes[i-1])).collect();
            layers.push(layer);
        }
        BrainFusionNet { layers }
    }

    /// Runs a forward pass and returns the output activations.
    pub fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        let activations = self.forward_with_cache(inputs);
        activations.last().cloned().unwrap_or_default()
    }

    fn forward_with_cache(&mut self, inputs: &[f64]) -> Vec<Vec<f64>> {
        if self.layers.is_empty() {
            return Vec::new();
        }
        if inputs.len() != self.layers[0].get(0).map_or(0, |neuron| neuron.weights.len()) {
            return Vec::new();
        }
        let mut current = inputs.to_vec();
        let mut activations = Vec::with_capacity(self.layers.len());
        for layer in self.layers.iter_mut() {
            let mut next = Vec::with_capacity(layer.len());
            for neuron in layer.iter_mut() {
                next.push(neuron.activate(&current));
            }
            activations.push(next.clone());
            current = next;
        }
        activations
    }

    /// Applies backpropagation updates for the provided targets and learning rate.
    pub fn backpropagate(&mut self, inputs: &[f64], targets: &[f64], learning_rate: f64) {
        let activations = self.forward_with_cache(inputs);
        let outputs = activations.last().cloned().unwrap_or_default();
        if outputs.is_empty() || outputs.len() != targets.len() {
            return;
        }

        let output_layer = self.layers.len() - 1;
        for (i, neuron) in self.layers[output_layer].iter_mut().enumerate() {
            let error = targets[i] - outputs[i];
            neuron.delta = error * outputs[i] * (SIGMOID_ONE - outputs[i]);
        }

        for l in (0..output_layer).rev() {
            for (i, neuron) in self.layers[l].iter_mut().enumerate() {
                let error: f64 = self.layers[l+1].iter().map(|n| n.delta * n.weights[i]).sum();
                neuron.delta = error * neuron.activation * (SIGMOID_ONE - neuron.activation);
            }
        }

        for (l, layer) in self.layers.iter_mut().enumerate() {
            let prev = if l == 0 { inputs } else { &activations[l-1] };
            for neuron in layer.iter_mut() {
                for (w, &input) in neuron.weights.iter_mut().zip(prev.iter()) {
                    *w += learning_rate * neuron.delta * input;
                }
                neuron.bias += learning_rate * neuron.delta;
            }
        }
    }

    /// Runs a Hebbian learning update across all layers.
    pub fn train_hebbian(&mut self, inputs: &[f64], learning_rate: f64) {
        let activations = self.forward_with_cache(inputs);
        if activations.is_empty() && !self.layers.is_empty() {
            return;
        }
        for (l, layer) in self.layers.iter_mut().enumerate() {
            let prev = if l == 0 { inputs } else { &activations[l-1] };
            for neuron in layer.iter_mut() {
                neuron.hebbian_update(prev, learning_rate);
            }
        }
    }
}
