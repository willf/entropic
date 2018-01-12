require 'entropic/version'

# Public: classes and methods useful for estimating entropy on strings.
#
# Examples
#
#   model = Entropic.Model.read("ngrams.tsv")
#   model.predict("the")
#   # => { log_prob_total: -101.1, log_prob_average: -20.02, size: 5 }
#
module Entropic
  # Public: create a sliding window of ngrams from a string
  #
  # string: The String to slide over
  # n: The Integer defining the size of the ngrams
  #
  # Examples
  #
  #  sliding('01234', 2)
  #  # => ['01', '12', '23', '34']
  #
  def self.sliding(string, n)
    (0..string.length - n).map { |i| (string[i, n]).to_s }
  end

  # Public: a counter for ngrams
  class NGramCounter
    attr_accessor :size, :counts, :total
    def initialize(size)
      @size = size
      @counts = Hash.new(0)
      @total = 0
    end

    # Public: update a counter with a string, and a multiplier
    #
    # Examples
    #
    # counter = NGramCounter.new(2)
    # counter.update_with_multiplier('01234', 1)
    #
    # string: The String to update with
    # multiplier: The Integer describing how much weight (will often be 1)
    #
    def update_with_multiplier(string, multiplier)
      Entropic.sliding(string, @size).each do |ngram|
        @counts[ngram] += multiplier
        @total += multiplier
      end
    end

    # Public: update a counter with a string, with a multiplier of 1
    #
    # Examples
    #
    # counter = NGramCounter.new(2)
    # counter.update('01234')
    #
    # string: The String to update with
    #
    def update(string)
      update_with_multiplier(string, 1)
    end

    # Public: get count for string, with default
    #
    # Examples
    #
    # counter = NGramCounter.new(2)
    # counter.update('01234')
    # counter.count('01', 0)
    # #=> 1
    # counter.count('bob, 0)
    # #=> 0
    #
    # ngram: The String to check
    # if_not_found : what to update with
    #
    def count(ngram, if_not_found)
      @counts.fetch(ngram, if_not_found)
    end
  end

  # Public; A model for entropy
  class Model
    VERSION = '1.0.0'.freeze
    attr_accessor :size, :counter

    def initialize(size)
      @size = size
      @counter = NGramCounter.new(size)
    end

    # Public: update a model with a string, and a multiplier
    #
    # Examples
    #
    # model = Model.new(2)
    # model.update_with_multiplier('01234', 1)
    #
    # string: The String to update with
    # multiplier: The Integer describing how much weight (will often be 1)
    #
    def update_with_multiplier(string, multiplier)
      @counter.update_with_multiplier(string, multiplier)
    end

    # Public: update a model with a string, with mulitplier or 1
    #
    # Examples
    #
    # model = Model.new(2)
    # model.update('01234')
    #
    # string: The String to update with
    #
    def update(string)
      update_with_multiplier(string, 1)
    end

    # Public: log probability of a ngram string in a model
    # returns value of first suffix of string
    # or log_prob of a 1-gram appearing once if no suffix found
    #
    # Examples
    #
    # model = Model.new(2)
    # model.update('01234')
    # model.log_prob('01')
    #
    # string: The String to query
    #
    def log_prob(key)
      if @counter.total == 0 || !key || key == ''
        return Math.log(0, 2.0) # -Infinity
      end

      count = @counter.count(key, 0.5)
      Math.log(count, 2.0) - Math.log(counter.total, 2.0)
    end

    # Public: dump model to some io object
    #
    # io: the IOWriter to write to
    #
    def dump(io)
      @counter.counts.each do |ngram, count|
        io.write("#{@size}\t#{ngram}\t#{count}\n")
      end
    end

    # Public: predict the log_prob sum and average over a string
    #         which will be split into ngrams
    #
    # string: The String to query
    #
    # returns: a dictionary of
    #  - log_prob_total
    #  - log_prob_average
    #  - size (number of ngrams in string)
    def predict(string)
      ngrams = Entropic.sliding(string, @size)
      log_prob_total = ngrams.map { |ngram| log_prob(ngram) }.inject(0.0, :+)
      log_prob_average = log_prob_total / ngrams.size.to_f
      { log_prob_total: log_prob_total, log_prob_average: log_prob_average, size: ngrams.size }
    end

    # Public: predict the entropy over a string
    #         which will be split into ngrams
    #
    # string: The String to query
    # 
    # returns: entropy
    def entropy(string)
      -predict(string)[:log_prob_average]
    end

    # Public: create a Model from reading from an IO object
    #
    # io: the IOReader
    #
    # returns: Model with stats filled in, and size of largest ngram
    def self.read(io)
      model = nil
      io.each_line do |string|
        ngram_size, ngram, count = string.strip.split(/\t/)
        ngram_size = ngram_size.to_i
        count = count.to_f
        model = Model.new(ngram_size) unless model
        counter = model.counter
        counter.total += count
        counter.counts[ngram] = count
      end
      model
    end

    # Public: Train a model on a bunch of data, line by line
    #
    # io: the IOReader
    #
    def train(io)
      io.each_line do |string|
        update(string.strip)
      end
    end

    # Public: Train a model on a bunch of data, line by line,
    #         with a multiplier
    #         each data line should be <string><tab><multiplier>
    #
    # io: the IOReader
    #
    def train_with_multiplier(io)
      io.each_line do |string|
        text, count = string.strip.split(/\t/)
        count = count.to_i
        update_with_multiplier(text, count)
      end
    end
  end
end
