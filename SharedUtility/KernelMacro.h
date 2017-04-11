#ifndef KERNELMACRO_H_
#define KERNELMACRO_H_

#define GLOBAL_TID() (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x

#endif /*KERNELMACRO_H_*/
