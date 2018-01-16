# Entropic

Entropic trains and predicts entropy of strings based on character n-gram models. For example:

```ruby
require 'entropic'
>> m = Entropic::Model.read(open('https://raw.githubusercontent.com/willf/entropy/master/data/google_books_2.tsv')); true
=> true
>> m.entropy("entropy")
=> 10.15243685946792
>> m.entropy("yportne")
=> 11.048928592721346
```

The string 'yportne' is much less likely than the string 'entropy'.

You can also train a model, using strings one per line.

```ruby
>> n = Entropic::Model.new(2); true
=> true
>> File.open('/tmp/training.txt') {|f| n.train(f)}; true
=> true
>> n.entropy('love')
=> 5.132072254636385
```

You can also train a model, using strings and a count of the number of times it appers, tab separated.

```ruby
>> o = Entropic::Model.new(2); true
=> true
>> File.open('/tmp/training_with_counts.txt') {|f| o.train_with_multiplier(f)}; true
=> true
>> o.entropy('love')
=> 5.132072254636385
```

You can also dump a model, to be read later.

```ruby
>> File.open('/tmp/save.tsv','w') {|f| o.dump(f)}; true
=> true
```

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'entropic'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install entropic


## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and tags, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/willf/entropic.

