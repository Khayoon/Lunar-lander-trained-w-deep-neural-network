using Flux

# model declaration activate with relu
function lunar_lander_sprite()
    model = Chain(
        Dense(8, 32, relu),
        Dense(32, 16, relu),
        Dense(16, 4)
    )
    return model
end
