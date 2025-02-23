use burn::tensor::{backend::AutodiffBackend, Data, Tensor};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{Adam, AdamConfig, Optimizer};
use burn_ndarray::NdArray;  // ✅ Correct import from burn-ndarray
use rand::Rng;
use textplots::{Chart, Plot, Shape};

#[derive(Module, Debug)]
struct LinearRegression<B: AutodiffBackend> {
    layer: Linear<B>,
}

impl<B: AutodiffBackend> LinearRegression<B> {
    fn new() -> Self {
        Self {
            layer: LinearConfig::new(1, 1).init(),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.layer.forward(x)
    }

    fn loss(y_pred: Tensor<B, 2>, y_true: Tensor<B, 2>) -> Tensor<B, 1> {
        let diff = y_pred - y_true;
        (diff.clone() * diff).mean() // Mean Squared Error (MSE)
    }
}

// Generate synthetic training data
fn generate_synthetic_data<B: AutodiffBackend>(n: usize) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let mut rng = rand::thread_rng();
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for _ in 0..n {
        let x: f32 = rng.gen_range(-10.0..10.0); // Random x value
        let noise: f32 = rng.gen_range(-1.0..1.0); // Random noise
        let y = 2.0 * x + 1.0 + noise; // y = 2x + 1 + noise
        x_data.push([x]);
        y_data.push([y]);
    }

    let x_tensor = Tensor::from_data(Data::from(x_data));
    let y_tensor = Tensor::from_data(Data::from(y_data));

    (x_tensor, y_tensor)
}

// Train the model
fn train_model<B: AutodiffBackend>() {
    let device = B::default();
    let mut model = LinearRegression::<B>::new();  // ✅ Model needs to be mutable for optimization

    // ✅ Proper optimizer initialization (linked to model parameters)
    let optim_config = AdamConfig::new();
    let mut optimizer = Adam::new(&optim_config, &model);

    let (x_train, y_train) = generate_synthetic_data::<B>(100);

    let mut loss_history = Vec::new();

    for epoch in 0..100 {
        let y_pred = model.forward(x_train.clone());
        let loss = LinearRegression::<B>::loss(y_pred, y_train.clone());

        // ✅ Corrected optimizer step
        optimizer.step(&mut model, &loss);

        let loss_value: f32 = loss.into_scalar();
        loss_history.push((epoch as f32, loss_value));

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss_value);
        }
    }

    // Plot training loss
    println!("\nTraining Loss Curve:");
    Chart::new(100, 20, 0.0, 100.0)
        .lineplot(&Shape::Lines(&loss_history))
        .display();
}

fn main() {
    train_model::<NdArray>(); // ✅ Corrected backend usage
}
