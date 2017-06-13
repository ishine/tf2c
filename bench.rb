#!/usr/bin/ruby

tests = Dir.glob('tests/*_large.py').sort

tests.each do |test|
  name = test[/tests\/(.*)_large\.py$/, 1]
  full = name + '_large'
  tf = `python runtest.py bench #{full} 2> /dev/null`.to_f
  c = `out/#{full}.exe --bench`.to_f
  puts "#{name} %f %f %.2f%%" % [tf, c, (c/tf*100)]
end
