#!/usr/bin/ruby

if ARGV[0]
  names = ARGV
else
  tests = Dir.glob('tests/*_large.py').sort
  names = tests.map do |test|
    test[/tests\/(.*)_large\.py$/, 1]
  end
end

names.each do |name|
  full = name + '_large'
  tf = `python runtest.py bench #{full} 2> /dev/null`.to_f
  c = `out/#{full}.exe --bench`.to_f
  puts "#{name} %f %f %.2f%%" % [tf, c, (c/tf*100)]
end
