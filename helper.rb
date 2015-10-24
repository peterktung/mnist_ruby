module DeepLearning
  module Helper
    def self.sigmoid(z)
        one = NMatrix.new(z.shape, 1.0)
        return one / (one + (-z).exp)
    end

    def self.sigmoid_prime(z)
        one = NMatrix.new(z.shape, 1.0)
        sigmoid(z) * (one - sigmoid(z))
    end
  end
end
