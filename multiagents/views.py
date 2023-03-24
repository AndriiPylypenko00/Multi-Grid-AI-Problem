from django.shortcuts import render

from multiagents.components.bio import Evolutionary, Map
from multiagents.components.lib import *

from .forms import InputForm


def get_index(request):
    form = InputForm(request.POST or None)
    if request.method == "POST":
        if form.is_valid():
            world = Map(
                size_x=form.cleaned_data["size_x"],
                size_y=form.cleaned_data["size_y"],
                num_of_cookies=form.cleaned_data["num_of_cookies"],
                cookie_prize=form.cleaned_data["cookie_prize"],
                move_punishment=form.cleaned_data["move_punishment"],
                border_punishment=form.cleaned_data["border_punishment"],
            )
            evolution = Evolutionary(
                num_layers=form.cleaned_data["num_of_layers"],
                num_inputs=24,
                num_outputs=8,
                num_of_agents=form.cleaned_data["num_of_agents"],
                mutation_chance=1 - form.cleaned_data["mutation_prob"],
                num_of_neurons=form.cleaned_data["num_of_neurons"],
            )

            epochs = form.cleaned_data["num_of_epochs"]
            epochs_per_save = form.cleaned_data["epochs_per_save"]
            number_of_episodes = form.cleaned_data["number_of_episodes"]
            epochs_worlds = []
            is_completed = False
            for epoch in range(epochs):
                for episode in range(number_of_episodes):
                    for id in range(len(evolution.agents)):
                        evolution.agents[id].isdead = 0
                        evolution.agents[id].is_completed = 0
                        evolution.agents[id].score_calculated = False
                    cookies = world.reset_world()
                    for agent_idx in range(len(evolution.agents)):
                        world.generate_agent_pos(
                            evolution.agents[agent_idx], evolution.agents[agent_idx].color
                        )
                    if epoch % epochs_per_save == 0 and episode == number_of_episodes-1:
                        world_states = [world.map.copy().tolist()]
                    for _ in range(form.cleaned_data["num_moves"]):
                        for id, agent in enumerate(evolution.agents):
                            if not agent.isdead and not agent.is_completed:
                                agent_decision = agent.feed_forward(
                                    [
                                        *calculate_agents(agent.x, agent.y, world.map),
                                        *calculate_cookies(agent.x, agent.y, world.map),
                                        *calculate_walls(agent.x, agent.y, world.map),
                                    ]
                                )
                                cookies = world.move(
                                    int(np.argmax(agent_decision)), evolution.agents[id], cookies
                                )
                                agent.is_dead()
                                if agent.is_completed:
                                    is_completed = True
                                    break

                            elif not agent.score_calculated:
                                agent.calculate_score()
                        if epoch % epochs_per_save == 0 and episode == number_of_episodes-1:
                            world_states.append(world.map.copy().tolist())
                        if is_completed:
                            print("COMPLETED")
                            break
                    if is_completed:
                        break
                    for agent in evolution.agents:
                        if not agent.score_calculated:
                            agent.calculate_score()
                if epoch % epochs_per_save == 0:
                    epochs_worlds.append(world_states.copy())
                if is_completed:
                    break
                evolution.crossover()
                print(
                    f"EPOCH: {epoch}, BEST AGENT SCORE: {evolution.last_best.score}, NUMBER OF AGENTS: {len(evolution.agents)}"
                )
            return render(
                request,
                "main.html",
                {
                    "form": form,
                    "epochs_worlds": epochs_worlds,
                    "train_done": "disabled",
                    "size_x": list(range(form.cleaned_data["size_x"])),
                    "size_y": list(range(form.cleaned_data["size_y"])),
                    "epochs": list(range(len(epochs_worlds))),
                    "moves": list(range(len(epochs_worlds[0]))),
                    "max_epochs": len(epochs_worlds),
                    "max_moves": len(epochs_worlds[0]),
                },
            )
    return render(request, "main.html", {"form": form, "train_done": False})
