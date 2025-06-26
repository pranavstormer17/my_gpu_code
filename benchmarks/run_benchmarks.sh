echo "kernel,config,time_ms" > results.csv

nvcc ../vector_add/vector_add.cu -o vecAdd
for threads in 64 128 256 512; do
  time_ms=$(./vecAdd $threads | awk '{print $NF}')
  echo "vecAdd,threads=$threads,$time_ms" >> results.csv
done

nvcc ../matrix_mul/matrix_mul.cu -o matMul
time_ms=$(./matMul | awk -F'[:,]' '{print $2}' )  # captures first time
echo "matMul,naive,$time_ms" >> results.csv
time_ms=$(./matMul | awk -F'[:,]' '{print $3}' )  # captures second time
echo "matMul,tiled,$time_ms" >> results.csv

echo "Done. See results.csv"
