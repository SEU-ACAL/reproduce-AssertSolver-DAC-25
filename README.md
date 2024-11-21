# AssertSolver

Open-source repo for "Insights from Rights and Wrongs: A Large Language Model for Solving Assertion Failures in RTL Design"



## Directory Structure

- `testbench/`:   Testbench we open-sourced

## Model Link

The model we open-sourced can be downloaded from [AssertSolver](https://huggingface.co/1412312anonymous/AssertSolver).


## Install 

`pip install llamafactory`

## Usage

`llamafactory cli assertsolver_infer_cli.yaml `

### Input Format

```
There is a systemverilog code that contains a bug, and will trigger assertions during formal verification:
<your code with bug>

The formal verification output was:
<assert log from formal verification tool>

The specification file of this code is: 
<your spec>

Please return me a json to tell me how the code should be modified, in the following format: '{"buggy_code": "The buggy code in the systemverilog (just one line of code)", "correct_code": "The correct code (just one line of code that can directly replace the buggy code, without any other description)"}'.
Ensure the response without any description like "```json```" and can be parsed by Python json.loads.
```