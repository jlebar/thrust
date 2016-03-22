/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/system/cuda/detail/bulk/detail/config.hpp>
#include <thrust/system/cuda/detail/bulk/execution_policy.hpp>
#include <thrust/system/cuda/detail/bulk/malloc.hpp>
#include <thrust/system/cuda/detail/bulk/algorithm/copy.hpp>
#include <thrust/system/cuda/detail/bulk/algorithm/accumulate.hpp>
#include <thrust/system/cuda/detail/bulk/uninitialized.hpp>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/function_traits.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>


BULK_NAMESPACE_PREFIX
namespace bulk
{


template<std::size_t bound, std::size_t grainsize, typename RandomAccessIterator1, typename RandomAccessIterator2, typename T, typename BinaryFunction>
__device__
__attribute__((noinline))
RandomAccessIterator2
  inclusive_scan(const bounded<bound, bulk::agent<grainsize> > &exec,
                 RandomAccessIterator1 first,
                 RandomAccessIterator1 last,
                 RandomAccessIterator2 result,
                 T init,
                 BinaryFunction binary_op)
{
  for(int i = 0; i < exec.bound(); ++i)
  {
    if(first + i < last)
    {
      init = binary_op(init, first[i]);
      result[i] = init;
    } // end if
  } // end for

  return result + (last - first);
} // end inclusive_scan


template<std::size_t bound, std::size_t grainsize, typename RandomAccessIterator1, typename RandomAccessIterator2, typename T, typename BinaryFunction>
__device__
__attribute__((noinline))
RandomAccessIterator2
  exclusive_scan(const bounded<bound, bulk::agent<grainsize> > &exec,
                 RandomAccessIterator1 first,
                 RandomAccessIterator1 last,
                 RandomAccessIterator2 result,
                 T init,
                 BinaryFunction binary_op)
{
  for(int i = 0; i < exec.bound(); ++i)
  {
    if(first + i < last)
    {
      result[i] = init;
      init = binary_op(init, first[i]);
    } // end if
  } // end for

  return result + (last - first);
} // end exclusive_scan


namespace detail
{
namespace scan_detail
{


template<typename InputIterator, typename OutputIterator, typename BinaryFunction>
struct scan_intermediate
  : thrust::detail::eval_if<
      thrust::detail::has_result_type<BinaryFunction>::value,
      thrust::detail::result_type<BinaryFunction>,
      thrust::detail::eval_if<
        thrust::detail::is_output_iterator<OutputIterator>::value,
        thrust::iterator_value<InputIterator>,
        thrust::iterator_value<OutputIterator>
      >
    >
{};


template<typename ConcurrentGroup, typename RandomAccessIterator, typename T, typename BinaryFunction>
inline __device__ T __attribute__((always_inline)) inplace_exclusive_scan(ConcurrentGroup &g, RandomAccessIterator first, T init, BinaryFunction binary_op)
{
  auto tid = g.this_exec.index();

  auto* p = first;
  if(tid == 0) p[0] = 42;
#ifdef __CUDA_ARCH__
  __syncthreads();
#endif
  if (tid == 32 && p[0] != 42) asm("trap;");
  return T();
}

template<typename ConcurrentGroup, typename RandomAccessIterator, typename Size, typename T, typename BinaryFunction>
inline __device__ T __attribute__((always_inline)) small_inplace_exclusive_scan(ConcurrentGroup &g, RandomAccessIterator first, Size n, T init, BinaryFunction binary_op)
{
  auto tid = g.this_exec.index();
  //asm ("");

  if (tid == 0) first[0] = 42;
  g.wait();
  return T();
}

// the upper bound on n is g.size().
template<typename ConcurrentGroup, typename RandomAccessIterator, typename Size, typename T, typename BinaryFunction>
__device__ T
#ifdef INLINE
inline __attribute__((always_inline))
#else
__attribute__((noinline))
#endif
bounded_inplace_exclusive_scan(ConcurrentGroup &g, RandomAccessIterator first, Size n, T init, BinaryFunction binary_op)
{
  return n == g.size()
             ? inplace_exclusive_scan(g, first, init, binary_op)
             : small_inplace_exclusive_scan(g, first, n, init, binary_op);
}

template<bool inclusive,
         std::size_t bound, std::size_t groupsize, std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename T,
         typename BinaryFunction>
__device__
__attribute__((noinline))
T
scan(bulk::bounded<
       bound,
       bulk::concurrent_group<bulk::agent<grainsize>,groupsize>
     > &g,
     RandomAccessIterator1 first, RandomAccessIterator1 last,
     RandomAccessIterator2 result,
     T carry_in,
     BinaryFunction binary_op)
{
  auto n = last - first;
  const auto spine_n = (n >= g.size() * g.this_exec.grainsize()) ? g.size() : (n + g.this_exec.grainsize() - 1) / g.this_exec.grainsize();
  bounded_inplace_exclusive_scan(g, result, spine_n, carry_in, binary_op);
  return T();
}


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename RandomAccessIterator2, typename BinaryFunction>
struct scan_buffer
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type  input_type;

  typedef typename scan_intermediate<
    RandomAccessIterator1,
    RandomAccessIterator2,
    BinaryFunction
  >::type intermediate_type;

  union
  {
    uninitialized_array<input_type, groupsize * grainsize>        inputs;
    uninitialized_array<intermediate_type, groupsize * grainsize> results;
  };
};


template<bool inclusive, std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename RandomAccessIterator2, typename T, typename BinaryFunction>
__device__ void scan_with_buffer(bulk::concurrent_group<bulk::agent<grainsize>,groupsize> &g,
                                 RandomAccessIterator1 first, RandomAccessIterator1 last,
                                 RandomAccessIterator2 result,
                                 T carry_in,
                                 BinaryFunction binary_op,
                                 scan_buffer<groupsize,grainsize,RandomAccessIterator1,RandomAccessIterator2,BinaryFunction> &buffer)
{
  typedef scan_buffer<
    groupsize,
    grainsize,
    RandomAccessIterator1,
    RandomAccessIterator2,
    BinaryFunction
  > buffer_type;

  typedef typename buffer_type::input_type        input_type;
  typedef typename buffer_type::intermediate_type intermediate_type;

  // XXX grabbing this pointer up front before the loop is noticeably
  //     faster than dereferencing inputs or results inside buffer
  //     in the loop below
  union {
    input_type        *inputs;
    intermediate_type *results;
  } stage;

  stage.inputs = buffer.inputs.data();

  typedef typename bulk::concurrent_group<bulk::agent<grainsize>,groupsize>::size_type size_type;

  size_type tid = g.this_exec.index();

  const size_type elements_per_group = groupsize * grainsize;

  for(; first < last; first += elements_per_group, result += elements_per_group)
  {
    size_type partition_size = thrust::min<size_type>(elements_per_group, last - first);
    
    // stage data through shared memory
    bulk::copy_n(g, first, partition_size, stage.inputs);

    carry_in = scan<inclusive>(bulk::bound<elements_per_group>(g),
                               stage.inputs, stage.inputs + partition_size,
                               stage.results,
                               carry_in,
                               binary_op);
    
    // copy to result 
    bulk::copy_n(g, stage.results, partition_size, result);
  } // end for
} // end scan_with_buffer()


} // end scan_detail
} // end detail


template<std::size_t bound,
         std::size_t groupsize,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename T,
         typename BinaryFunction>
__device__
typename thrust::detail::enable_if<
  bound <= groupsize * grainsize,
  RandomAccessIterator2
>::type
__attribute__((noinline))
inclusive_scan(bulk::bounded<
                 bound,
                 bulk::concurrent_group<bulk::agent<grainsize>,groupsize>
               > &g,
               RandomAccessIterator1 first, RandomAccessIterator1 last,
               RandomAccessIterator2 result,
               T carry_in,
               BinaryFunction binary_op)
{
  detail::scan_detail::scan<true>(g, first, last, result, carry_in, binary_op);
  return result + (last - first);
} // end inclusive_scan()


template<std::size_t bound,
         std::size_t groupsize,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename BinaryFunction>
__device__
typename thrust::detail::enable_if<
  bound <= groupsize * grainsize,
  RandomAccessIterator2
>::type
__attribute__((noinline))
inclusive_scan(bulk::bounded<
                 bound,
                 bulk::concurrent_group<bulk::agent<grainsize>,groupsize>
               > &g,
               RandomAccessIterator1 first, RandomAccessIterator1 last,
               RandomAccessIterator2 result,
               BinaryFunction binary_op)
{
  if(bound > 0 && first < last)
  {
    typename thrust::iterator_value<RandomAccessIterator1>::type init = *first;

    // we need to wait because first may be the same as result
    g.wait();

    if(g.this_exec.index() == 0)
    {
      *result = init;
    }

    detail::scan_detail::scan<true>(g, first + 1, last, result + 1, init, binary_op);
  }

  return result + (last - first);
} // end inclusive_scan()


template<std::size_t groupsize,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename T,
         typename BinaryFunction>
__attribute__((noinline))
__device__ void inclusive_scan(bulk::concurrent_group<bulk::agent<grainsize>,groupsize> &g,
                               RandomAccessIterator1 first, RandomAccessIterator1 last,
                               RandomAccessIterator2 result,
                               T init,
                               BinaryFunction binary_op)
{
  typedef detail::scan_detail::scan_buffer<groupsize,grainsize,RandomAccessIterator1,RandomAccessIterator2,BinaryFunction> buffer_type;

#if __CUDA_ARCH__ >= 200
  buffer_type *buffer = reinterpret_cast<buffer_type*>(bulk::malloc(g, sizeof(buffer_type)));

  if(bulk::is_on_chip(buffer))
  {
    detail::scan_detail::scan_with_buffer<true>(g, first, last, result, init, binary_op, *bulk::on_chip_cast(buffer));
  } // end if
  else
  {
    detail::scan_detail::scan_with_buffer<true>(g, first, last, result, init, binary_op, *buffer);
  } // end else

  bulk::free(g, buffer);
#else
  __shared__ uninitialized<buffer_type> buffer;
  detail::scan_detail::scan_with_buffer<true>(g, first, last, result, init, binary_op, buffer.get());
#endif // __CUDA_ARCH__
} // end inclusive_scan()


template<std::size_t size,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename BinaryFunction>
__device__
RandomAccessIterator2
__attribute__((noinline))
inclusive_scan(bulk::concurrent_group<bulk::agent<grainsize>,size> &this_group,
               RandomAccessIterator1 first,
               RandomAccessIterator1 last,
               RandomAccessIterator2 result,
               BinaryFunction binary_op)
{
  if(first < last)
  {
    // the first input becomes the init
    // XXX convert to the immediate type when passing init to respect Thrust's semantics
    //     when Thrust adopts the semantics of N3724, just forward along *first
    //typename thrust::iterator_value<RandomAccessIterator1>::type init = *first;
    typename detail::scan_detail::scan_intermediate<
      RandomAccessIterator1,
      RandomAccessIterator2,
      BinaryFunction
    >::type init = *first;

    // we need to wait because first may be the same as result
    this_group.wait();

    if(this_group.this_exec.index() == 0)
    {
      *result = init;
    } // end if

    bulk::inclusive_scan(this_group, first + 1, last, result + 1, init, binary_op);
  } // end if

  return result + (last - first);
} // end inclusive_scan()


template<std::size_t bound, std::size_t groupsize, std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename T,
         typename BinaryFunction>
__device__
__attribute__((noinline))
typename thrust::detail::enable_if<
  bound <= groupsize * grainsize,
  RandomAccessIterator2
>::type
exclusive_scan(bulk::bounded<
                 bound,
                 bulk::concurrent_group<bulk::agent<grainsize>,groupsize>
               > &g,
               RandomAccessIterator1 first, RandomAccessIterator1 last,
               RandomAccessIterator2 result,
               T carry_in,
               BinaryFunction binary_op)
{
  detail::scan_detail::scan<true>(g, first, last, result, carry_in, binary_op);
  return result + (last - first);
} // end exclusive_scan()


template<std::size_t groupsize,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename T,
         typename BinaryFunction>
__device__
__attribute__((noinline))
typename thrust::detail::enable_if<
  (groupsize > 0),
  RandomAccessIterator2
>::type
exclusive_scan(bulk::concurrent_group<agent<grainsize>,groupsize> &g,
               RandomAccessIterator1 first, RandomAccessIterator1 last,
               RandomAccessIterator2 result,
               T init,
               BinaryFunction binary_op)
{
  typedef detail::scan_detail::scan_buffer<groupsize,grainsize,RandomAccessIterator1,RandomAccessIterator2,BinaryFunction> buffer_type;

#if __CUDA_ARCH__ >= 200
  buffer_type *buffer = reinterpret_cast<buffer_type*>(bulk::malloc(g, sizeof(buffer_type)));

  if(bulk::is_on_chip(buffer))
  {
    detail::scan_detail::scan_with_buffer<false>(g, first, last, result, init, binary_op, *bulk::on_chip_cast(buffer));
  } // end if
  else
  {
    detail::scan_detail::scan_with_buffer<false>(g, first, last, result, init, binary_op, *buffer);
  } // end else

  bulk::free(g, buffer);
#else
  __shared__ uninitialized<buffer_type> buffer;
  detail::scan_detail::scan_with_buffer<false>(g, first, last, result, init, binary_op, buffer.get());
#endif

  return result + (last - first);
} // end exclusive_scan()


} // end bulk
BULK_NAMESPACE_SUFFIX

