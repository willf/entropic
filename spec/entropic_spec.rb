require 'spec_helper'

RSpec.describe Entropic do
  it 'has a version number' do
    Entropic::VERSION.should be_a_kind_of String
  end

  describe '#sliding' do
    it 'should convert a string into an array of character ngrams' do
      Entropic.sliding('test', 2).should eq(%w(te es st))
    end
  end
end

RSpec.describe NGramCounter do
  model = NGramCounter.new(2)

  describe '::initialize' do
    it 'should allow a size to be specified' do
      model.size.should eq(2)
    end

    it 'should initialize total to 0' do
      model.total.should eq(0)
    end

    it 'initialize counts to an empty hash' do
      model.counts.size.should eq(0)
    end

    it 'should update the model when requested' do
      m = NGramCounter.new(2)
      m.update('test')
      m.total.should be > 0.0
      m.counts['te'].should be > 0.0
    end

    it 'should return 1.0 for values not found' do
      model.count('foo', 1.0).should eq(1.0)
    end
  end
end

RSpec.describe Model do
  se = Model.new(2)

  describe '::initialize' do
    it 'should allow a size to be specified' do
      se.size.should eq(2)
    end

    it 'should create a model dictionary of size maxsize' do
      se.counter.size.should eq(2)
    end
  end

  describe '#log_prob' do
    it 'should produce -Infinity when untrained' do
      se.log_prob('test').should eq(Math.log(0, 2.0))
    end

    it 'should produce a result when trained and ngram present' do
      se = Model.new(2)
      se.train(StringIO.new('test'))
      se.log_prob('te').should be < 0.0
      se.log_prob('te').should be > Math.log(0, 2.0)
    end

    it 'should produce a result when trained and ngram is not present' do
      se = Model.new(2)
      se.train(StringIO.new('testtest')) # te occurs 2 times; tt occurs 1
      se.log_prob('ZZ').should be < se.log_prob('te') # ZZ occurs 0 times, count as 0.5
      se.log_prob('ZZ').should < se.log_prob('tt')
    end
  end

  describe '#dump' do
    it 'should be able to read dumped data' do
      data = <<HERE
2	fa	13798.000000
2	PG	13.000000
2	Os	35.000000
2	zw	893.000000
2	6D	7.000000
2	Hg	1.000000
2	RU	15.000000
2	Zg	3.000000
2	tp	2237.000000
HERE
      se0 = Model.read(StringIO.new(data))
      se0.log_prob('fa').should be < 0.0
      se0.log_prob('fa').should be > Math.log(0, 2.0)
    end
  end

  describe '#predict' do
    it 'should be able to predict' do
      se0 = Model.new(2)
      data = <<HERE
2	fa	13798.000000
2	PG	13.000000
2	Os	35.000000
2	zw	893.000000
2	6D	7.000000
2	Hg	1.000000
2	RU	15.000000
2	Zg	3.000000
2	tp	2237.000000
HERE
      se0 = Model.read(StringIO.new(data))
      se0.size.should eq(2)
      key = 'faPGzw'
      d = se0.predict(key)
      d.fetch(:size, 0).should eq(5)
      d.fetch(:log_prob_total, 0).should be < 0.0
      d.fetch(:log_prob_total, 0).should_not be -Float::INFINITY
      d.fetch(:log_prob_average, 0).should be < 0.0
      d.fetch(:log_prob_average, 0).should_not be -Float::INFINITY
    end
  end

  describe '#entropy' do
    it 'should be able to calculate entropy' do
      se0 = Model.new(2)
      data = <<HERE
2	fa	13798.000000
2	PG	13.000000
2	Os	35.000000
2	zw	893.000000
2	6D	7.000000
2	Hg	1.000000
2	RU	15.000000
2	Zg	3.000000
2	tp	2237.000000
HERE
      se0 = Model.read(StringIO.new(data))
      se0.size.should eq(2)
      key = 'faPGzw'
      e = se0.entropy(key)
      e.should be > 0.0
      e.should_not be Float::INFINITY
    end
  end

  describe '#read' do
    it 'should be able to dump and read' do
      se0 = Model.new(2)
      se0.train(StringIO.new("test\nbest\n"))
      o = StringIO.new('', 'w')
      se0.dump(o)
      o.close
      se2 = Model.read(StringIO.new(o.string))
      key = 'faPGzw'
      ngrams = Entropic.sliding(key, 2)
      d = se2.predict(key)
      d.fetch(:size, 0).should eq(ngrams.size)
      d.fetch(:log_prob_total, 0).should be < 0.0
      d.fetch(:log_prob_total, 0).should_not be -Float::INFINITY
      d.fetch(:log_prob_average, 0).should be < 0.0
      d.fetch(:log_prob_average, 0).should_not be -Float::INFINITY
    end
  end
end
