#!/bin/sh

model=dynamicra
input_dir=synthetics
today=$(date +%Y%m%d)/

COMMAND="uv run python src/main.py --multirun \
  model=${model} \
  io.input_dir=${input_dir} \
  io.root_out_dir=out/${today} \
  data.n=1000 \
  data.d=10 \
  data.T=7 \
  data.K=2 \
  data.model_type=nonlinear \
  model.n_forward_mc=128 \
  model.regmodel=ols \
  model.transmodel=var \
  eval.variance_type=moment \
  save=True \
  verbose=True"

bash bin/run_wrapper.sh "$@" "$COMMAND"
