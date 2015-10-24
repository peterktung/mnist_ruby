module DeepLearning
    class Network
        #sizes is an array of int that defines the number of neurons in each layer
        def initialize(sizes)
            @num_layers = sizes.length
            @sizes = sizes
            @biases = []
            @weights = []
            
            rng = GSL::Rng.alloc
            @sizes[1..-1].each do |size|
                @biases << NMatrix.new([size,1], rng.gaussian(1, size).to_a)
            end

            # each value of @weights is an two array of two dimensional matrice
            # Ex: @weights[0][1,2] is the weight of the second neuron of layer '0'
            # (input layer) to the third neuron
            @sizes[0..-2].zip(@sizes[1..-1]).each do |x, y|
                @weights << NMatrix.new([y,x], rng.gaussian(1, x * y).to_a)
            end
        end

        def feedforward(a)
            output = a
            @biases.zip(@weights).each do |b, w|
                output = Helper.sigmoid(w.dot(output) + b)
            end

            output
        end

        #eta is the learning rate
        def sgd(training_data, epochs, mini_batch_size, eta, test_data=nil)
            n = training_data.length
            n_test = test_data.nil? ? 0: test_data.length 
            epochs.times do |i|
                training_data.shuffle!
                (0..n-1).step(mini_batch_size) do |k|
                    end_index = k + mini_batch_size - 1
                    mini_batch = end_index > n - 1 ? nil : training_data[k..end_index]
                    if mini_batch
                        update_mini_batch(mini_batch, eta)    
                    end
                end

                if test_data
                    puts "Epoch #{i+1}: #{evaluate(test_data)}/#{n_test}"
                else
                    puts "Epoch #{i+1} complete"
                end
            end
        end

        #Update the weights and biases using backpropagation based on one
        # mini batch
        def update_mini_batch(mini_batch, eta)
            nabla_b = []
            nabla_w = []
            @biases.each {|b| nabla_b << NMatrix.zeros(b.shape)}
            @weights.each {|w| nabla_w << NMatrix.zeros(w.shape)}

            mini_batch.each do |x, y|
                delta_nabla_b, delta_nabla_w = backprop(x,y)
                new_nabla_b = []
                new_nabla_w = []
                nabla_b.zip(delta_nabla_b).each {|nb, dnb| new_nabla_b << nb + dnb}
                nabla_w.zip(delta_nabla_w).each {|nw, dnw| new_nabla_w << nw + dnw}
                nabla_b = new_nabla_b
                nabla_w = new_nabla_w
            end

            #Core of SGD, averaging the gradient descent of the weights and biases 
            # of the mini batch, and moving the weight and bias in that direction
            # at the speed of 'eta'
            new_weights = []
            new_biases = []
            @weights.zip(nabla_w).each {|w, nw| new_weights << w - (nw / mini_batch.size) * eta}
            @biases.zip(nabla_b).each {|b, nb| new_biases << b - (nb / mini_batch.size) * eta}
            @weights = new_weights
            @biases = new_biases
        end
 
        def backprop(x, y)
            nabla_b = []
            nabla_w = []
            @biases.each {|b| nabla_b << NMatrix.zeros(b.shape)}
            @weights.each {|w| nabla_w << NMatrix.zeros(w.shape)}

            #calculates all the weighted inputs (z) activations layer by layer
            activation = x
            activations = [x]
            zs = []
            @biases.zip(@weights) do |b, w|
                z = w.dot(activation) + b
                zs << z
                activation = Helper.sigmoid(z)
                activations << activation
            end

            #Calcuates error of each neuron (delta) layer by layer starting with
            # with the last layer, working backwards
            delta = cost_derivative(activations[-1], y) * \
                Helper.sigmoid_prime(zs[-1])  #equation BP1a of chapter 2
            nabla_b[-1] = delta #equation BP3 of chapter 2
            nabla_w[-1] = delta.dot(activations[-2].transpose) #equation BP4 of chapter 2
            -2.downto(-@num_layers+1) do |l|
                z = zs[l]
                sp = Helper.sigmoid_prime(z)
                delta = @weights[l+1].transpose.dot(delta) * sp #equation BP2 of chapter 2
                nabla_b[l] = delta
                nabla_w[l] = delta.dot(activations[l-1].transpose)
            end

            return nabla_b, nabla_w
        end

        def evaluate(test_data)
            num_correct = 0
            test_data.each do |x, y|
                output = feedforward(x).to_a
                result = output.index(output.max)
                num_correct += 1 if result == y
            end

            num_correct
        end

        def cost_derivative(output_activations, y)
            new_y = NMatrix.zeros([10,1])
            new_y[y,0] = 1.0
            output_activations - new_y
        end
    end
end
