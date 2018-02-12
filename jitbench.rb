#!/usr/bin/env ruby

benchs = %w(matmul_large matmul_bias_large)

perfs = {}
benchs.each do |name|
  perfs[[name, true]] = []
  perfs[[name, false]] = []
end

benchs.each do |name|
  `python runtest.py output #{name} 2> /dev/null`
end

N = 3

N.times do
  benchs.each do |name|
    [false, true].each do |use_jit|
      r = `python runtest.py bench #{name} 2> /dev/null`
      gflops = r[/(\d+\.\d+) GFLOPS/, 1]
      if !gflops
        raise "No flops: #{r}"
      end
      perfs[[name, use_jit]] << gflops.to_f

      puts "#{name}#{use_jit ? "-jit" : ""} #{gflops} GFLOPS"
    end
  end
end

benchs.each do |name|
  jit = perfs[[name, true]].max
  nojit = perfs[[name, false]].max
  puts "#{name} #{nojit} #{jit}"
end
