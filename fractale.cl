inline uint computeIndex1D(uint2 index2D, uint2 size){
  return (uint)(index2D.y * size.x + index2D.x);
}

inline float2 complexNumber(uint2 index2D, uint2 size, float2 ll, float2 ur){
  return (float2)((ll.x + ((float)index2D.x)/(((float)size.x)-1.0)*(ur.x-ll.x)),
    (ll.y + ((float)index2D.y)/(((float)size.y)-1.0)*(ur.y-ll.y)));
}

inline uint divergence(float2 z, float2 c, uint max){
  uint n;
  uint div;
  for (n=0;n<max;n++) {
    div = n;
    if ((z.x*z.x + z.y*z.y) > 4.0) break;

    z = (float2)((z.x*z.x - z.y*z.y),(z.x*z.y + z.y*z.x)) + c;
  }
  return div;
}

__kernel void julia(__global uint *julia, uint2 size, float2 ll, float2 ur, float2 c, uint max) {
  // get the index of the current element to be processed
  uint2 index2D = (uint2)(get_global_id(0), get_global_id(1));

  // compute the 1D index
  uint index1D = computeIndex1D(index2D, size);

  // compute z from index2D
  float2 z = complexNumber(index2D, size, ll, ur);

  // compute the element, if we have to (avoid threads to write outside from allocated memory)
  if(index2D.x < size.x && index2D.y < size.y) {
    julia[index1D] = divergence(z, c, max);
  }
}
