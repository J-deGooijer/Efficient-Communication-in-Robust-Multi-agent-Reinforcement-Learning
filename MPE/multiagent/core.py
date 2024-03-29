import numpy as np
from MPC.car import Car
from MPC.controller import MPC, PID, SIMPLE_CONTROLLER

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 100.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # Car object
        self.car = Car(0,0)
        # MPC object
        self.horizon = 4
        self.controller = MPC(self.horizon)
        # self.controller = PID(horizon=self.horizon)
        self.controller = SIMPLE_CONTROLLER(self.horizon)

# multi-agent world
class World(object):
    def __init__(self, environment):
        self.env = environment
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.25
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 2e+2
        self.contact_margin = 2e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self, obs=None):
        if self.env == 'simple_tag_elisa':
            self.update_agent_elisa()           

        else:
            if self.env=='simple_tag_webots':
                if obs==None:
                    print('Warning, you indicated that the observation should come from webots but did not provide one')
                else:
                    self.set_webots_obs(obs)

            # set actions for scripted agents 
            for agent in self.scripted_agents:
                agent.action = agent.action_callback(agent, self)
            # gather forces applied to entities
            p_force = [None] * len(self.entities)
            # apply agent physical controls
            p_force = self.apply_action_force(p_force)
            # apply environment forces
            p_force = self.apply_environment_force(p_force)
            # integrate physical state
            self.integrate_state(p_force)
            # update agent state
            for agent in self.agents:
                self.update_agent_state(agent)
            
            if self.env == 'simple_tag_mpc':
                self.update_agent_mpc()


    def update_agent_mpc(self):      
        for idx, agent in enumerate(self.agents):
            waypoints = [agent.state.p_pos.tolist()] * agent.controller.horizon
            # waypoints = [[-0.5,-0.5]] * agent.controller.horizon
            for i in range(agent.controller.horizon):
                x, _, _ = agent.car.get_state()
                # linear_v, angular_v = agent.controller.optimize(car=agent.car, points=waypoints, dt=self.dt/agent.controller.horizon)
                # linear_v, angular_v = agent.controller.get_control_inputs(x, waypoints[i], agent.car.get_points()[2])
                linear_v, angular_v = agent.controller.get_control_inputs(x, waypoints[i])
                agent.car.set_robot_velocity(linear_v, angular_v)
                agent.car.update(self.dt/agent.controller.horizon)
            x, _, p_vel = agent.car.get_state()
            agent.state.p_pos = np.array([x[0, 0],x[1, 0]])
            agent.state.p_vel = np.array([p_vel[0][0],p_vel[1][0]])
                
            for p in range(self.dim_p):
                agent.state.p_pos[p] = self.bound(agent.state.p_pos[p])

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise                
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

            for p in range(self.dim_p):
                entity.state.p_pos[p] = self.bound(entity.state.p_pos[p])

    def update_agent_elisa(self):
        wheelbase = 0.04
        wheel_radius = 0.0042

        for agent in self.agents:
            arc = self.dt*wheel_radius*agent.action.u
            arc_avg = sum(arc)/len(arc)

            beta = (arc[0]-arc[1])/wheelbase
            if abs(beta)<0.01: # Gives a divide by zero error, robot goes straight
                agent.state.p_pos[0] += arc_avg*np.cos(agent.state.p_rot)
                agent.state.p_pos[1] += arc_avg*np.sin(agent.state.p_rot)
            else:
                R = arc_avg/beta
                x_rel = R*np.cos(beta)-R
                y_rel = R*np.sin(beta)
                agent.state.p_pos[0] += -np.sin(agent.state.p_rot)*x_rel + np.cos(agent.state.p_rot)*y_rel
                agent.state.p_pos[1] += np.cos(agent.state.p_rot)*x_rel + np.sin(agent.state.p_rot)*y_rel
                agent.state.p_rot += beta
            

            agent.state.p_vel = (arc_avg/self.dt)*np.array([np.cos(agent.state.p_rot[0]), np.sin(agent.state.p_rot[0])])

            for p in range(self.dim_p):
                agent.state.p_pos[p] = self.bound(agent.state.p_pos[p])

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k

        if dist != 0 and not np.isnan(delta_pos.any()) and not np.isnan(penetration):
            force = self.contact_force * delta_pos / dist * penetration
        else:
            force = 0  

        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
    

    def bound(self, x):
        if x < 1.0 and x > -1.0:
            return x
        elif x >= 1.0:
            return 1.0
        elif x <= -1.0:
            return -1.0
        
    def set_webots_obs(self, obs):
        for i, agent in enumerate(self.agents):
            agent.state.p_vel = obs[i][0:2]
            agent.state.p_pos = obs[i][2:4]