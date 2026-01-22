use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

const DEFAULT_MEMBRANE_POTENTIAL: f64 = 0.0;
const DEFAULT_THRESHOLD: f64 = 1.0;
const DEFAULT_RESET_POTENTIAL: f64 = 0.0;
const DEFAULT_TAU: f64 = 20.0;
const DEFAULT_REFRACTORY_PERIOD: f64 = 5.0;
const DEFAULT_LAST_SPIKE_TIME: f64 = -1.0;
const DEFAULT_A_PLUS: f64 = 0.01;
const DEFAULT_A_MINUS: f64 = -0.012;
const DEFAULT_TAU_PLUS: f64 = 20.0;
const DEFAULT_TAU_MINUS: f64 = 20.0;
const DEFAULT_LEARNING_RATE: f64 = 0.001;
const DEFAULT_MIN_WEIGHT: f64 = -1.0;
const DEFAULT_MAX_WEIGHT: f64 = 1.0;
const DEFAULT_WEIGHT_MIN: f64 = -0.1;
const DEFAULT_WEIGHT_MAX: f64 = 0.1;
const INPUT_SPIKE_MAGNITUDE: f64 = 1.0;

#[derive(Debug, Clone)]
pub struct LifNeuron {
    membrane_potential: f64,
    threshold: f64,
    reset_potential: f64,
    tau: f64,
    refractory_period: f64,
    time_since_spike: f64,
    last_spike_time: f64,
    spiked: bool,
}

impl LifNeuron {
    /// Creates a leaky integrate-and-fire neuron with default parameters.
    pub fn new() -> Self {
        LifNeuron {
            membrane_potential: DEFAULT_MEMBRANE_POTENTIAL,
            threshold: DEFAULT_THRESHOLD,
            reset_potential: DEFAULT_RESET_POTENTIAL,
            tau: DEFAULT_TAU,
            refractory_period: DEFAULT_REFRACTORY_PERIOD,
            time_since_spike: 0.0,
            last_spike_time: DEFAULT_LAST_SPIKE_TIME,
            spiked: false,
        }
    }

    /// Integrates the neuron state over a time step and updates spike status.
    pub fn integrate(&mut self, input_current: f64, dt: f64, current_time: f64) {
        self.time_since_spike += dt;
        if self.time_since_spike < self.refractory_period {
            self.membrane_potential = self.reset_potential;
            self.spiked = false;
            return;
        }
        self.membrane_potential += dt * (-self.membrane_potential / self.tau + input_current);
        self.spiked = if self.membrane_potential >= self.threshold {
            self.membrane_potential = self.reset_potential;
            self.time_since_spike = 0.0;
            self.last_spike_time = current_time;
            true
        } else {
            false
        };
    }
}

#[derive(Debug, Clone)]
pub struct SpikingNet {
    neurons: Vec<LifNeuron>,
    weights: Vec<Vec<f64>>,
    a_plus: f64,
    a_minus: f64,
    tau_plus: f64,
    tau_minus: f64,
    learning_rate: f64,
    min_weight: f64,
    max_weight: f64,
    sim_time: f64,
}

impl SpikingNet {
    /// Builds a fully connected spiking network with randomized weights.
    pub fn new(num_neurons: usize) -> Self {
        let mut rng = thread_rng();
        let uniform = Uniform::from(DEFAULT_WEIGHT_MIN..DEFAULT_WEIGHT_MAX);
        let weights: Vec<Vec<f64>> = (0..num_neurons).map(|_| (0..num_neurons).map(|_| uniform.sample(&mut rng)).collect()).collect();
        let neurons = (0..num_neurons).map(|_| LifNeuron::new()).collect();
        SpikingNet {
            neurons,
            weights,
            a_plus: DEFAULT_A_PLUS,
            a_minus: DEFAULT_A_MINUS,
            tau_plus: DEFAULT_TAU_PLUS,
            tau_minus: DEFAULT_TAU_MINUS,
            learning_rate: DEFAULT_LEARNING_RATE,
            min_weight: DEFAULT_MIN_WEIGHT,
            max_weight: DEFAULT_MAX_WEIGHT,
            sim_time: 0.0,
        }
    }

    /// Advances the network by a time step and returns neuron spike events.
    pub fn step(&mut self, inputs: &[f64], dt: f64) -> Vec<bool> {
        let neuron_count = self.neurons.len();
        self.sim_time += dt;
        let mut currents = vec![0.0; neuron_count];
        for (index, &input) in inputs.iter().enumerate().take(neuron_count) {
            currents[index] = input;
        }
        for i in 0..neuron_count {
            for j in 0..neuron_count {
                if self.neurons[j].spiked {
                    currents[i] += self.weights[i][j];
                }
            }
        }
        let mut spikes = vec![false; neuron_count];
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            neuron.integrate(currents[i], dt, self.sim_time);
            spikes[i] = neuron.spiked;
        }
        self.apply_stdp();
        spikes
    }

    fn apply_stdp(&mut self) {
        for post_idx in 0..self.neurons.len() {
            for pre_idx in 0..self.neurons.len() {
                let post_time = self.neurons[post_idx].last_spike_time;
                let pre_time = self.neurons[pre_idx].last_spike_time;
                if post_time < 0.0 || pre_time < 0.0 {
                    continue;
                }
                let delta_t = post_time - pre_time;
                let delta_w = if delta_t > 0.0 {
                    self.a_plus * (-delta_t / self.tau_plus).exp()
                } else {
                    self.a_minus * (delta_t / self.tau_minus).exp()
                };
                let updated = self.weights[post_idx][pre_idx] + self.learning_rate * delta_w;
                self.weights[post_idx][pre_idx] = updated.clamp(self.min_weight, self.max_weight);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoihiEmulator {
    net: SpikingNet,
    pending_spikes: Vec<(usize, f64)>,
}

impl LoihiEmulator {
    /// Creates a Loihi-style emulator backed by a spiking network.
    pub fn new(num_neurons: usize) -> Self {
        Self {
            net: SpikingNet::new(num_neurons),
            pending_spikes: Vec::new(),
        }
    }

    /// Queues an external spike for injection into the next simulation step.
    pub fn inject_spike(&mut self, neuron_idx: usize, time: f64) {
        self.pending_spikes.push((neuron_idx, time));
    }

    /// Processes queued spikes and advances the network by one time step.
    pub fn step_event(&mut self, dt: f64) -> Vec<bool> {
        let mut inputs = vec![0.0; self.net.neurons.len()];
        for (neuron_idx, _) in self.pending_spikes.drain(..) {
            if let Some(input) = inputs.get_mut(neuron_idx) {
                *input += INPUT_SPIKE_MAGNITUDE;
            }
        }
        self.net.step(&inputs, dt)
    }

    /// Returns an immutable view of the underlying spiking network.
    pub fn network(&self) -> &SpikingNet {
        &self.net
    }
}
