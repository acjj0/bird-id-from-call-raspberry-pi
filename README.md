# bird-id-from-call-raspberry-pi

Note: Before copying code, please check with author acjj0

Identifying Bird Species From Their Sounds On Raspberry Pi
- Identifies microphone hardware, opens a stream, and converts live audio into chunks
- Loads tflite model generated by training on sounds from [Cornell Lab of Ornithology](https://www.birds.cornell.edu/home/) 
- Deteremines neural network's expected dimensions for input and output, then adjusts chunks into the expected dimensions
- Allocates tensors, sets input tensors, invokes the interpreter, and derives the resultant tensors
- Interprets results into classes for bird species identification, rank orders by highest confidence, and writes to Postgres Database
