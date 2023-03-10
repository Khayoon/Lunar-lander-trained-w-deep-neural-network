using Flux
using Gym
using JLD2

include("model.jl")

function train_lunar_lander(env, model, optimizer, epochs; file_path="")
    if !isempty(file_path)
        # load trained model parameters
        trained_params = load(file_path, "params")
        Flux.loadparams!(model, trained_params)
    end
    
    for epoch in 1:epochs
        total_reward = 0.0
        state = env.reset()
        done = false
        while !done
            action_probs = Flux.softmax(model(state))
            action = sample(1:4, action_probs)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            loss = crossentropy(action_probs, onehot(action, 4))
            grad = gradient(() -> loss, Flux.params(model))
            Flux.update!(optimizer, Flux.params(model), grad)
            state = next_state
        end
        @show epoch, total_reward
        
        if !isempty(file_path)
            # save model parameters after each epoch
            trained_params = Flux.params(model)
            save(file_path, "params", trained_params)
        end
    end
end

function load_lunar_lander_model(file_path)
    trained_params = load(file_path, "params")
    model = lunar_lander_model()
    Flux.loadparams!(model, trained_params)
    return model
end

env = Gym.make("LunarLander-v2")
model = lunar_lander_model()
optimizer = ADAM(0.001)
epochs = 100

train_lunar_lander(env, model, optimizer, epochs)

# save trained model parameters
trained_params = Flux.params(model)
save("trained_model.jld2", "params", trained_params)

# load trained model parameters
loaded_model = load_lunar_lander_model("trained_model.jld2")
