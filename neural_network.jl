using Random

# Layer for Neural Network
struct Layer
    __num__::Int
    __activ__::String
end

Layer(num::Int) = Layer(num, "relu")

# Defines the structure of Neural Network
struct NeuralNet
    input_dims::Array{Int}
    layers::Array{Layer}
    loss_func::String
    __weights__::Array{Array{Float64}}
    __biases__::Array{Array{Float64}}
end

# Builds the Neural Network
function NeuralNet(input_dims::Array{Int}, layers::Layer...; loss_func="cross_entropy")
    l0 = prod(input_dims)
    weights = Array{Array{Float64}}(undef, 0)
    biases = Array{Array{Float64}}(undef, 0)
    ls = Array{Layer}(undef, 0)
    for layer in layers
        push!(weights, rand_normal(layer.__num__, l0, stddev=0.05))
        push!(biases, 0.5*ones(layer.__num__, 1))
        push!(ls, layer)
        l0 = layer.__num__
    end
    return NeuralNet(input_dims, ls, loss_func, weights, biases)
end

# Normal distribution with mean = 0 and given Standard deviation
function rand_normal(dims::Int...; stddev::Float64=1.0)
    if stddev <= 0.0
        error("Standard deviation must be positive")
    end
    u1 = rand(dims...)
    u2 = rand(dims...)
    r = @. sqrt(-2.0*log(u1))
    theta = 2.0*pi*u2
    return @. stddev*(r*sin(theta))
end

# Relu activation Function
function relu(z)
    return @. max(0, z)
end

# Derivative of Relu Function
function derelu(z)
    return [x >= 0 ? 1 : 0 for x in z]
end

# Leaky Relu activation Function
function leaky_relu(z)
    return @. max(0.01*z, z)
end

# Derivative of Leaky Relu Function
function deleaky_relu(z)
    return [x >= 0 ? 1 : 0.01 for x in z]
end

# Sigmoid activation Function
function sigmoid(z)
    return @. 1 / (1 + exp(-z))
end

# Derivative of Sigmoid Function
function desigmoid(a)
    return @. a*(1 - a)
end

# Derivative of Tanh Function
function detanh(a)
    return @. 1 - a*a
end

# Softmax Function
function softmax(z)
    exps = @. exp(z)
    return exps / sum(exps)
end

# Derivative of Softmax Function
function desoftmax(p)
    perr = zeros(10, 10)
    for i = 1:10
        for j = 1:10
            if i == j
                perr[i, j] = p[i]*(1 - p[j])
            else
                perr[i, j] = -p[i]*p[j]
            end
        end
    end
    return perr
end

# Calls the activation function
function activate(z, activ="relu")
    if activ == "relu"
        return relu(z)
    elseif activ == "leaky_relu"
        return leaky_relu(z)
    elseif activ == "sigmoid"
        return sigmoid(z)
    elseif activ == "tanh"
        return tanh.(z)
    elseif activ == "softmax"
        return softmax(z)
    end
end

# Derivative of activation function
function deactivate(z, a, activ="relu")
    if activ == "relu"
        return derelu(z)
    elseif activ == "leaky_relu"
        return deleaky_relu(z)
    elseif activ == "sigmoid"
        return desigmoid(a)
    elseif activ == "tanh"
        return detanh.(a)
    elseif activ == "softmax"
        return desoftmax(a)
    end
end

# Cross Entropy Function
function cross_entropy(pred, y)
    return -sum(@. y*log([p > 1e-10 ? p : 1 for p in pred]))
end

# Derivative of Cross Entropy Function
function decross_entropy(pred, y)
    return @. -y/pred
end

# Loss Function
function loss_function(pred, y, loss_func="cross_entropy")
    if loss_func == "cross_entropy"
        return cross_entropy(pred, y)
    end
end

# Derivative of Loss Function
function deloss_function(pred, y, loss_func="cross_entropy")
    if loss_func == "cross_entropy"
        return decross_entropy(pred, y)
    end
end

# Forward Propagation for Neural Network
function forward_propagation(net::NeuralNet, x::Array{Float64})
    z = Array{Float64}[]
    a = Array{Float64}[]
    push!(a, x)
    for (w, b, l) in zip(net.__weights__, net.__biases__, net.layers)
        push!(z, w*a[end] + b)
        push!(a, activate(z[end], l.__activ__))
    end
    return z, a
end

# Backward Propagation for Neural Network
function back_propagation!(net::NeuralNet, y::Array{Float64}, z::Array{Array{Float64}}, a::Array{Array{Float64}}; learning_rate=0.01)
    da = deloss_function(a[end], y)
    l = length(z)
    for i in l:-1:1
        if net.layers[i].__activ__ == "softmax"
            dz = deactivate(z[i], a[i + 1], net.layers[i].__activ__) * da
        else
            dz = @. da * deactivate(z[i], a[i + 1], net.layers[i].__activ__)
        end
        dw = dz * transpose(a[i])
        db = dz
        da = transpose(net.__weights__[i]) * dz
        net.__weights__[i] -= learning_rate * dw
        if size(dw) != size(net.__weights__[i])
            println(size(dw), ' ', size(net.__weights__[i]))
            error("Something has gone terribly wrong!")
        end
        net.__biases__[i] -= learning_rate * db
    end
end

# Read images from file
function read_image(path)
    println("Reading Images...")
    bytes = open(path) do file
        read(file)
    end
    number_of_images = bytes[5] * 256^3 + bytes[6] * 256^2 + bytes[7] * 256 + bytes[8]
    rows = Int(bytes[12])
    cols = Int(bytes[16])
    img_size = rows * cols
    println("Number of images = ", number_of_images, " rows = ", rows, " cols = ", cols, " Image size = ", img_size)
    images = zeros(img_size, number_of_images)
    i = 1
    for byte in bytes[17:end]
        images[i] = byte / 255.0
        i += 1
    end
    println("Done.")
    return images
end

# Read labels from file
function read_label(path)
    bytes = open(path) do file
        read(file)
    end
    number_of_labels = bytes[5] * 256^3 + bytes[6] * 256^2 + bytes[7] * 256 + bytes[8]
    println("Number of labels = ", number_of_labels)
    labels = zeros(10, number_of_labels)
    i = 1
    for byte in bytes[9:end]
        labels[byte + 1, i] = 1.0
        i += 1
    end
    println("Done.")
    return labels
end

# Read all the image and label files
function read_data()
    x_train = read_image("train-images.idx3-ubyte")
    y_train = read_label("train-labels.idx1-ubyte")
    x_test = read_image("t10k-images.idx3-ubyte")
    y_test = read_label("t10k-labels.idx1-ubyte")
    return x_train, y_train, x_test, y_test
end

# Returns the index of the maximum value
function max_index(arr)
    max_val = 0
    index = 0
    for i = 1:length(arr)
        if arr[i] > max_val
            max_val = arr[i]
            index = i
        end
    end
    return index
end

# Creating and training neural net
function main()
    Random.seed!(0)
    x_train, y_train, x_test, y_test = read_data()
    model = NeuralNet([784, 1], Layer(20), Layer(10, "softmax"))
    for i = 1:60000
        x, y = x_train[1:end, i], y_train[1:end, i]
        z, a = forward_propagation(model, x)
        back_propagation!(model, y, z, a, learning_rate=0.0095)
        if i % 5000 == 0
            println("loss = ", loss_function(a[end], y))
        end
    end
    correct = 0
    for i = 1:10000
        x, y = x_test[1:end, i], y_test[1:end, i]
        z, a = forward_propagation(model, x)
        correct += max_index(a[end]) == max_index(y)
    end
    println(correct / 100)
end

main()