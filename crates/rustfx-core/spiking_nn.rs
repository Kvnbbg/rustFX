// spiking_nn.rs - Simple Spiking Neural Network (LIF model)
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

#[derive(Debug, Clone)]
pub struct LifNeuron {
    membrane_potential: f64,
    threshold: f64,
    reset_potential: f64,
    tau: f64, // Time constant
    spiked: bool,
}

impl LifNeuron {
    pub fn new() -> Self {
        LifNeuron {
            membrane_potential: 0.0,
            threshold: 1.0,
            reset_potential: 0.0,
            tau: 20.0,
            spiked: false,
        }
    }

    pub fn integrate(&mut self, input_current: f64, dt: f64) {
        self.membrane_potential += dt * (-self.membrane_potential / self.tau + input_current);
        self.spiked = if self.membrane_potential >= self.threshold {
            self.membrane_potential = self.reset_potential;
            true
        } else {
            false
        };
    }
}

#[derive(Debug, Clone)]
pub struct SpikingNet {
    neurons: Vec<LifNeuron>,
    weights: Vec<Vec<f64>>, // Connectivity
}

impl SpikingNet {
    pub fn new(num_neurons: usize) -> Self {
        let mut rng = thread_rng();
        let uniform = Uniform::from(-0.1..0.1);
        let weights: Vec<Vec<f64>> = (0..num_neurons).map(|_| (0..num_neurons).map(|_| uniform.sample(&mut rng)).collect()).collect();
        let neurons = (0..num_neurons).map(|_| LifNeuron::new()).collect();
        SpikingNet { neurons, weights }
    }

    pub fn step(&mut self, inputs: &[f64], dt: f64) -> Vec<bool> {
        let mut currents = inputs.to_vec();
        for i in 0..self.neurons.len() {
            for j in 0..self.neurons.len() {
                if self.neurons[j].spiked {
                    currents[i] += self.weights[i][j];
                }
            }
        }
        let mut spikes = vec![false; self.neurons.len()];
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            neuron.integrate(currents[i], dt);
            spikes[i] = neuron.spiked;
        }
        spikes
    }
}
