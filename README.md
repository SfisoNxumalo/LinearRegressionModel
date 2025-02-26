# Introduction

This project implements a simple AI model for linear regression using the Rust programming language and the Burn library. The goal is to predict the output of the function: y = 2x + 1. Using synthetic data for training. The model is trained using the Adam optimizer and the Mean Squared Error (MSE) loss function.

Setup and Installation

Before running the project, ensure you have the following installed:

- Rust (latest stable version)
- Cargo (Rust package manager)
- Steps to Set Up the Project
- Clone this repository:

```
git clone https://github.com/SfisoNxumalo/LinearRegressionModel.git
```

```
cd LinearRegressionModel
```

Add the necessary dependencies to Cargo.toml:

```
[dependencies]
rand = "0.9.0"
burn = { version = "0.16.0", features = ["wgpu", "train"] }
burn-ndarray = "0.16.0"
rgb = "0.8.5"
textplots = "0.8.6"
```
Build and run the project:

```
cargo run
```

### Approach
- Generating Synthetic Data
Created random (x, y) pairs where y = 2x + 1, adding noise to simulate real-world conditions.

- Defining the Model
Implemented a simple linear regression model using the Burn library. Defined a forward pass and used Mean Squared Error as the loss function.

- Training the Model

Tied to used the Adam optimizer to minimize the loss function and to monitored the training process by tracking the loss over epochs.

- Challenges Faced

Rust Language Barriers

Since this was my first time using Rust, I found it difficult to understand and debug errors. The strict type system and unfamiliar syntax made it challenging to implement the model correctly.

AI-Generated Code Issues

I used AI to assist in writing the code, but it sometimes generated incorrect solutions, such as suggesting non-existent functions or incorrect API calls. This resulted in compilation errors that were difficult to debug.

Burn Library Complexity.

The Burn library is still relatively new, and documentation is limited. Understanding how to properly set up tensors, initialize optimizers, and define a proper training loop required extensive research.

- Results & Evaluation

The project encountered multiple errors during compilation and execution. Due to limited experience with Rust and the Burn library, I was unable to fully resolve these errors. However, I gained a deeper understanding of Rust's strict type system and how AI-generated code can sometimes lead to incorrect implementations.

- Reflection on Learning

AI and Documentation Support

Relied heavily on AI to generate and troubleshoot code, but faced challenges due to incorrect suggestions. I consulted the official Burn library documentation and Rust forums to understand better approaches.

Key Takeaways

Rust requires a strong understanding of ownership, borrowing, and type inference. AI-generated code is helpful but should always be verified against official documentation. Debugging Rust errors is challenging, but it enhances problem-solving skills.

Future Improvements

Gain more hands-on experience with Rust by working on smaller projects first. Study the Burn library documentation in detail before attempting another ML project in Rust.

## Resources Used

Rust official documentation (https://doc.rust-lang.org/)

Burn library documentation (https://burn.dev/)

AI-generated code (ChatGPT, Deepseek)

Online forums and Rust community discussions