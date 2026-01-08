## Quadrotor Motion Primitives Generation

### 1. Build dynoplan

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="path/to/openrobots/" -DCMAKE_BUILD_TYPE=Release
make -j4
cd ..
```

### 2. Prepare Output Directory

```bash
mkdir quad3d_prims
```

### 3. Generate Primitives and Visualize

```bash
./gen_primitives.sh quad3d_v0 300 ../quad3d_prims/quad3d ../quad3d_prims/prims.html
```
- The first argument is the dynamics type (e.g., `quad3d_v0`).
- The second is the number of initial primitives.
- The third and fourth are the output and visualization file paths (should be **relative to the build folder**).

### 4. Adjusting Optimization Parameters
- You can change optimization parameters in `opt_params.yaml`.

### 5. Output
- After running, you will find an HTML file (e.g., `prims.html`) showing 10 random primitives.
- You can change the number of samples by editing the `num_samples` argument in the `.sh` file, but using a high number is not recommended as it may slow down or break the HTML visualization.
