import agentpy as ap
import networkx as nx
import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Definição da Mensagem FIPA-ACL
class FipaAclMessage:
    def __init__(self, performative, sender, receiver, content, conversation_id=None):
        self.performative = performative
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.conversation_id = conversation_id

    def __repr__(self):
        return f"FipaAclMessage({self.performative}, {self.sender}, {self.receiver}, {self.content}, {self.conversation_id})"

# Definição dos Agentes
class SocialAgent(ap.Agent):
    def setup(self):
        self.opinion = self.p.initial_opinion
        self.influence_factor = self.p.influence_factor
        self.inbox = []

    def send(self, message):
        receiver = self.model.agents[message.receiver]
        receiver.inbox.append(message)

    def receive(self):
        if self.inbox:
            return self.inbox.pop(0)
        return None

    def step(self):
        # Enviar pedido de opiniões para todos os vizinhos
        neighbors = list(self.network.neighbors(self))
        for neighbor in neighbors:
            message = FipaAclMessage(
                performative="request",
                sender=self.id,
                receiver=neighbor.id,
                content="opinion"
            )
            self.send(message)

        # Processar mensagens recebidas
        received_opinions = []
        while self.inbox:
            message = self.receive()
            if message.performative == "request" and message.content == "opinion":
                reply = FipaAclMessage(
                    performative="inform",
                    sender=self.id,
                    receiver=message.sender,
                    content=self.opinion
                )
                self.send(reply)
            elif message.performative == "inform":
                received_opinions.append(message.content)

        # Atualizar opinião com base nas opiniões recebidas
        if received_opinions:
            if self.type == 'Common':
                self.opinion = sum(received_opinions) / len(received_opinions)
            elif self.type == 'Influencer':
                self.opinion = max(set(received_opinions), key=received_opinions.count)
            elif self.type == 'Extremist':
                self.opinion = self.opinion  # Extremistas não mudam de opinião
            elif self.type == 'Moderate':
                mean_opinion = sum(received_opinions) / len(received_opinions)
                std_dev = np.std(received_opinions)
                self.opinion = (self.opinion + mean_opinion) / 2 + random.gauss(0, std_dev / 2)
            elif self.type == 'Conformist':
                self.opinion = np.median(received_opinions)
            elif self.type == 'Rebel':
                self.opinion = -sum(received_opinions) / len(received_opinions)
            elif self.type == 'OpinionLeader':
                self.opinion = random.choice(received_opinions)
            elif self.type == 'Sporadic':
                if random.random() < 0.8:
                    self.opinion = random.choice(received_opinions)

class OpinionModel(ap.Model):
    def setup(self):
        # Criar rede
        if self.p.network_type == 'random':
            graph = nx.erdos_renyi_graph(self.p.num_agents, self.p.connection_prob)
        elif self.p.network_type == 'scale_free':
            graph = nx.barabasi_albert_graph(self.p.num_agents, self.p.num_edges)
        elif self.p.network_type == 'small_world':
            graph = nx.watts_strogatz_graph(self.p.num_agents, self.p.num_neighbors, self.p.rewire_prob)
        elif self.p.network_type == 'ring':
            graph = nx.cycle_graph(self.p.num_agents)

        # Criar agentes
        self.agents = ap.AgentList(self, self.p.num_agents, SocialAgent)

        # Adicionar agentes à rede
        self.network = self.agents.network = ap.Network(self, graph)
        self.network.add_agents(self.agents, self.network.nodes)

        #Definir quantidades de ligações que tornam um agente do tipo influencer
        degrees = list(dict(graph.degree()).values())
        sorted_degrees = sorted(degrees, reverse=True)
        num_influencers = int(self.p.influencer_fraction * self.p.num_agents)
        influencers_degree = sorted_degrees[:num_influencers]

        #Selecionar os agentes influencers
        influencer_agents = []
        for agent in self.agents:
            neighbors = list(self.network.neighbors(agent))
            degree = len(neighbors)
            if degree in influencers_degree and len(influencers_degree) <= num_influencers:
                influencer_agents.append(agent)
    
        agent_types = ['Common'] * int(self.p.common_fraction * self.p.num_agents) + \
                      ['Extremist'] * int(self.p.extremist_fraction * self.p.num_agents) + \
                      ['Moderate'] * int(self.p.moderate_fraction * self.p.num_agents) + \
                      ['Conformist'] * int(self.p.conformist_fraction * self.p.num_agents) + \
                      ['Rebel'] * int(self.p.rebel_fraction * self.p.num_agents) + \
                      ['OpinionLeader'] * int(self.p.opinion_leader_fraction * self.p.num_agents) + \
                      ['Sporadic'] * int(self.p.sporadic_fraction * self.p.num_agents)
        random.shuffle(agent_types)
        
        other_agents = [agent for agent in list(self.agents) if agent not in influencer_agents]
        for agent, type in zip(other_agents, agent_types):
            agent.type = type
            agent.id = self.agents.index(agent)
        for agent in influencer_agents:
            agent.type = 'Influencer'
            agent.id = self.agents.index(agent)
        
    def step(self):
        self.agents.step()
    
    def update(self):
        self.record('opinions', [agent.opinion for agent in self.agents])
        self.record('types', [agent.type for agent in self.agents])
        self.record('networks', self.network.graph)
    
    def end(self):
        self.report('final_opinions', [agent.opinion for agent in self.agents])

# Parâmetros do modelo
parameters = {
    'initial_opinion': 0.5,
    'influence_factor': 0.6,
    'num_agents': 100,
    'common_fraction': 0.5,
    'influencer_fraction': 0.1,
    'extremist_fraction': 0.1,
    'moderate_fraction': 0.1,
    'conformist_fraction': 0.1,
    'rebel_fraction': 0.05,
    'opinion_leader_fraction': 0.03,
    'sporadic_fraction': 0.02,
    'network_type': 'ring',  # 'random', 'scale_free', 'small_world', 'ring'
    'connection_prob': 0.1,
    'num_edges': 2,
    'num_neighbors': 4,
    'rewire_prob': 0.1,
    'branch_factor': 2,
    'tree_depth': 3,
    'steps': 1000
}

# Executar o modelo
model = OpinionModel(parameters)
results = model.run()

# Visualizar a polarização e evolução das opiniões
opinion_data = results.variables.OpinionModel.opinions

# Calcular polarização
polarization = [np.std(step_opinions) for step_opinions in opinion_data]

# Plotar polarização ao longo do tempo
polarization_fig = px.line(x=list(range(len(polarization))), y=polarization, labels={'x': 'Passos de Tempo', 'y': 'Polarização (Desvio Padrão das Opiniões)'}, title='Polarização ao Longo do Tempo')
polarization_fig.write_html('C:\\Users\\Daniel\\Desktop\\figuras agentes inteligentes\\polarization.html')
polarization_fig.show()

# Criar um motion chart das opiniões
fig = go.Figure(
    data=[
        go.Histogram(x=opinion_data[0], nbinsx=10, name='Opiniões')
    ],
    layout=go.Layout(
        title='Evolução das Opiniões ao Longo do Tempo',
        xaxis=dict(range=[0, 1], title='Opinião'),
        yaxis=dict(range=[0, len(opinion_data[0]) / 2], title='Frequência'),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(label='Play',
                          method='animate',
                          args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)])])]
    ),
    frames=[go.Frame(data=[go.Histogram(x=opinions, nbinsx=10)], name=str(i)) for i, opinions in enumerate(opinion_data)]
)

fig.write_html('C:\\Users\\Daniel\\Desktop\\figuras agentes inteligentes\\opinions.html')
fig.show()

# Função para converter a rede para Plotly
def get_network_edges(G):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    return edge_x, edge_y

# Função para converter os agentes para Plotly
def get_network_nodes(G, opinions, types):
    node_x = []
    node_y = []
    node_color = []
    node_symbols = []
    type_to_symbol = {
        'Common': 'circle',
        'Influencer': 'star',
        'Extremist': 'diamond',
        'Moderate': 'square',
        'Conformist': 'triangle-up',
        'Rebel': 'x',
        'OpinionLeader': 'pentagon',
        'Sporadic': 'hexagon'
    }
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_color.append(opinions[list(G.nodes()).index(node)])
        node_symbols.append(type_to_symbol[types[list(G.nodes()).index(node)]])
    return node_x, node_y, node_color, node_symbols

# Gerar posições dos nós
pos = nx.spring_layout(results.variables.OpinionModel.networks[0])  # Posição fixa para consistência
for t, G in enumerate(results.variables.OpinionModel.networks):
    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]

# Criar frames para a animação da rede
frames = []
for t, G in enumerate(results.variables.OpinionModel.networks):
    edge_x, edge_y = get_network_edges(G)
    node_x, node_y, node_color, node_symbols = get_network_nodes(G, opinion_data[t], [agent.type for agent in model.agents])
    frames.append(go.Frame(data=[
        go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888')),
        go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            marker=dict(size=10, color=node_color, colorscale='Viridis', colorbar=dict(title='Opinião'), symbol=node_symbols)
        )
    ], name=str(t)))

# Criar figura inicial da animação da rede
edge_x, edge_y = get_network_edges(results.variables.OpinionModel.networks[0])
node_x, node_y, node_color, node_symbols = get_network_nodes(results.variables.OpinionModel.networks[0], opinion_data[0], [agent.type for agent in model.agents])
fig_network = go.Figure(
    data=[
        go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888')),
        go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            marker=dict(size=10, color=node_color, colorscale='Viridis', colorbar=dict(title='Opinião'), symbol=node_symbols),
            text=[f"Agente {i}, Tipo: {t}" for i, t in enumerate([agent.type for agent in model.agents])],
            hoverinfo='text',
            name='Agentes'
        )
    ],
    layout=go.Layout(
        title='Rede e Opiniões ao Longo do Tempo',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        legend=dict(title='Tipos de Agentes'),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(label='Play',
                          method='animate',
                          args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)])])]
    ),
    frames=frames
)

fig_network.write_html('C:\\Users\\Daniel\\Desktop\\figuras agentes inteligentes\\network.html')
fig_network.show()

# Criar dataframe para o gráfico de dispersão animado
data = []
for t, opinions in enumerate(opinion_data):
    for i, opinion in enumerate(opinions):
        data.append({'Time': t, 'Agent': i, 'Opinion': opinion, 'Type': model.agents[i].type})

df = pd.DataFrame(data)

# Criar gráfico de dispersão animado
fig_scatter = px.scatter(df, x='Agent', y='Opinion', animation_frame='Time', animation_group='Agent', color='Type',
                         title='Formação dos Grupos por Opinião ao Longo do Tempo',
                         labels={'Agent': 'Agente', 'Opinion': 'Opinião', 'Time': 'Tempo', 'Type': 'Tipo de Agente'})
fig_scatter.update_layout(xaxis={'categoryorder': 'category ascending'})

fig_scatter.write_html('C:\\Users\\Daniel\\Desktop\\figuras agentes inteligentes\\scatter.html')
fig_scatter.show()

# Impacto dos tipos de agentes
agent_types = ['Common', 'Influencer', 'Extremist', 'Moderate', 'Conformist', 'Rebel', 'OpinionLeader', 'Sporadic']
impact_data = {atype: [] for atype in agent_types}
for t in range(len(opinion_data)):
    for atype in agent_types:
        impact_data[atype].append(sum(opinion for opinion, agent_type in zip(opinion_data[t], results.variables.OpinionModel.types[t]) if agent_type == atype))

impact_fig = go.Figure()
for atype in agent_types:
    impact_fig.add_trace(go.Scatter(x=list(range(len(impact_data[atype]))), y=impact_data[atype], mode='lines', name=atype))

impact_fig.update_layout(title='Impacto dos Tipos de Agentes ao Longo do Tempo', xaxis_title='Passos de Tempo', yaxis_title='Impacto (Soma das Opiniões)')
impact_fig.write_html('C:\\Users\\Daniel\\Desktop\\figuras agentes inteligentes\\impact.html')
impact_fig.show()

#results.variables.OpinionModel.type
opinion_data = pd.DataFrame(opinion_data)
opinion_data.to_csv(r'C:\Users\Daniel\Desktop\figuras agentes inteligentes\dados.csv', index=False)

# Adicionar logging
print(f"Número de agentes comuns: {parameters['common_fraction'] * parameters['num_agents']}")
print(f"Número de influenciadores: {parameters['influencer_fraction'] * parameters['num_agents']}")
print(f"Número de extremistas: {parameters['extremist_fraction'] * parameters['num_agents']}")
print(f"Número de moderados: {parameters['moderate_fraction'] * parameters['num_agents']}")
print(f"Número de conformistas: {parameters['conformist_fraction'] * parameters['num_agents']}")
print(f"Número de rebeldes: {parameters['rebel_fraction'] * parameters['num_agents']}")
print(f"Número de líderes de opinião: {parameters['opinion_leader_fraction'] * parameters['num_agents']}")
print(f"Número de agentes esporádicos: {parameters['sporadic_fraction'] * parameters['num_agents']}")