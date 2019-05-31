import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding




MAX_NUM_STEPS = np.inf


FPS    = 50
SCALE  = 4   # affects how fast-paced the game is, forces should be adjusted as well


MAIN_ENGINE_POWER  = 13.0
SIDE_ENGINE_POWER  =  6

INITIAL_RANDOM = 1000.0  
ROCKET_POLY =[
	(-12,-6), (-12,6), (+6,6),  (+12,0), (6,-6)	
	]


VIEWPORT_W = 900
VIEWPORT_H = 600

class ContactDetector(contactListener):
	def __init__(self, env):
		contactListener.__init__(self)
		self.env = env

		#revise below
	def BeginContact(self, contact):
		if self.env.rocket in [contact.fixtureA.body,contact.fixtureB.body]:
			if self.env.planet in [contact.fixtureA.body,contact.fixtureB.body]:
				self.env.success = True
			else:
				self.env.game_over = True

class RealtimeRocketOpt(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : FPS
	}

	def __init__(self):
		self.seed()
		self.viewer = None

		self.world = Box2D.b2World()
#set zero gravity
		self.world.gravity = (0,0)

		self.planet = None
		self.rocket = None
		self.particles = []
		# self.obstacles = 

		self.observation_space = spaces.Box(-np.inf, np.inf, shape=(21,), dtype=np.float32)

		#assuming discrete actions
		self.action_space = spaces.Discrete(4)

		self.reset()

	def seed(self, seed = None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def torgb(self,a,b,c):
		return np.array([float(a)/255, float(b)/255, float(c)/255])

	def _destroy(self):
		if not self.planet: return
		self.world.contactListener = None
		self._clean_particles(True)
		self.world.DestroyBody(self.planet)
		self.planet = None
		self.world.DestroyBody(self.rocket)
		self.rocket = None

	def make_satellite(self, density, vel, rad, pos, res, color1 = (0,0,0), color2 = (0,0,0), speedup = 1):
		planet_poly = []
		for i in range(res):
			ang = -2*math.pi*i / res
			planet_poly.append((math.cos(ang)*rad/SCALE, math.sin(ang)*rad/SCALE ))
		fix = fixtureDef(shape = polygonShape(vertices = planet_poly), density = density, restitution=0.0)
		planet = self.world.CreateDynamicBody(position = pos, fixtures = fix)
		planet.color1 = self.torgb(color1[0],color1[1],color1[2])
		planet.color2 = self.torgb(color2[0],color2[1],color2[2])

		#applying initial speed
		vec = planet.worldCenter - self.planet.worldCenter
		G = 1e-5
		speedup = 600 * vec.length * speedup
		initial_mag = math.sqrt(G * self.planet.mass / vec.length) * speedup
		vel_vec = (Box2D.b2Vec2(vec[1],-vec[0]) / vec.length) * initial_mag 
		planet.linearVelocity = vel_vec
		return planet
		


	def reset(self):
		self._destroy()
		self.world.contactListener_keepref = ContactDetector(self)
		self.world.contactListener = self.world.contactListener_keepref
		self.game_over = False
		self.success = False
		self.prev_reward = 0
		self.initial_step = True

		W = VIEWPORT_W/SCALE
		H = VIEWPORT_H/SCALE

		
#backgrounds
		self.universe_poly = [(0,0), (0, VIEWPORT_H/SCALE), (VIEWPORT_W/SCALE, VIEWPORT_H/SCALE), (VIEWPORT_W/SCALE,0)]
		self.universe_color = self.torgb(14,24,32)
		self.star_poly_comp = [[  
							(0.0    ,    1.0),
							(0.22   ,    0.30),
							(-0.22  ,     0.30)],
							[
							(-0.95  ,     0.30),
							(0.95   ,    0.30),
							(0.36   ,   -0.11),
							(-0.36  ,    -0.11)],
							[
							(-0.36  ,    -0.11),
							(0.36   ,   -0.11),
							(-0.58  ,    -0.80)],
							[
							(0.0    ,   -0.38),
							(0.36   ,   -0.11),
							(0.58   ,   -0.80)
							]]
		self.star_poly_comp = np.array([np.array(i) for i in self.star_poly_comp]) * 2
		self.star_color1 = self.torgb(240,255,224)
#make random stars
		num_stars = 40
		self.random_x = np.random.uniform(0,VIEWPORT_W/SCALE,num_stars)
		self.random_y = np.random.uniform(0,VIEWPORT_H/SCALE,num_stars)
		self.random_loc = np.stack([self.random_x,self.random_y], axis = 1)

		self.stars=[]
		for i in range(len(self.random_loc)):
			for j in range(len(self.star_poly_comp)):
				self.stars.append(self.star_poly_comp[j] + self.random_loc[i])

		self.planet_pos = (0.9 * VIEWPORT_W/SCALE,0.9 * VIEWPORT_H/SCALE)

		self.planet_radius = 50
		res = 16
		self.planet_poly = []
		for i in range(res):
			ang = -2*math.pi*i / res
			self.planet_poly.append((math.cos(ang)*self.planet_radius/SCALE, math.sin(ang)*self.planet_radius/SCALE ))
		self.planet_density = 687 #Saturn's density
		fix = fixtureDef(shape = polygonShape(vertices = self.planet_poly), density = self.planet_density, restitution=0.0)
		self.planet = self.world.CreateDynamicBody(position = self.planet_pos, fixtures = fix)
		self.planet.color1 = self.torgb(123.0,120.0,105.0)
		self.planet.color2 = self.torgb(191.0,189.0,175.0)

		meta_titan = np.array([1.87, 3.57, 2574/5833 * 70, self.planet.position - (122/SCALE, 122/SCALE), 16,(0,0,0),(0,0,0)])
		meta_mimas = np.array([1.14, 14.28, 198/5833* 70, self.planet.position - (185/SCALE, 185/SCALE), 8, (0,0,0),(0,0,0)])
		meta_dione = np.array([1.478, 8.5, 560/5833* 70, self.planet.position - (377/SCALE, 377/SCALE), 5, (0,0,0),(0,0,0)])
		meta_data = [meta_titan,meta_mimas,meta_dione]
		self.satellites = []
		self.satellites_colors = []
		for satel in meta_data:
			self.satellites.append(self.make_satellite(*satel))

		self.rocket = self.world.CreateDynamicBody(
			position = (100/SCALE,100/SCALE),
			angle=0.0,
			fixtures = fixtureDef(
				shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in ROCKET_POLY]),
				density=1.0,
				# friction=0.1,
				categoryBits=0x0010,
				maskBits=0x001,  # collide only with planets and obstacles
				restitution=0.0))

		#to check engine loc
		self.temp_poly=[]
		self.temp_poly.append((math.cos(ang)*self.planet_radius*10 + self.rocket.position[0], math.sin(ang)*self.planet_radius*10 + self.rocket.position[1]))

		#need to adjust if you change the shape of rocket
		#distance between center of the rocket to main engine
		self.dis_to_main_engine = 8.4/SCALE
		self.dis_to_side_engine = 6/SCALE



		self.rocket.color1 = (0.2,0.3,0.3)
		self.rocket.color2 = (0.1,0.2,0.2)

		#create obstacles
		self.obstacles = []

		self.drawlist = [self.rocket] + self.obstacles
		#assume discrete case
		return self.step(0)[0]


	def attraction(self, body_a, body_b):
		#assume body_b has a lot bigger mass
		G = 1e-5
		loc_a = body_a.worldCenter
		loc_b = body_b.worldCenter
		force_vector = loc_b - loc_a
		force_vector /= force_vector.length
		mag = G * body_a.mass * body_b.mass /(force_vector.lengthSquared)
		return force_vector * mag

	def _create_particle(self, mass, x, y, ttl):
		p = self.world.CreateDynamicBody(
			position = (x,y),
			angle=0.0,
			fixtures = fixtureDef(
				shape=circleShape(radius=2/SCALE, pos=(0,0)),
				density=mass,
				friction=0.1,
				categoryBits=0x0100,
				maskBits=0x001,  # collide only with obs+planets
				restitution=0.3)
				)
		p.ttl = ttl
		self.particles.append(p)
		self._clean_particles(False)
		return p

	def _clean_particles(self, all):
		while self.particles and (all or self.particles[0].ttl<0):
			self.world.DestroyBody(self.particles.pop(0))

	def distance(self, pos_roc,pos_plan):
		square = math.pow(pos_roc.x - pos_plan[0], 2) + math.pow(pos_roc.y - pos_plan[1], 2)
		return math.sqrt(square)

	def step(self, action):

		#assume discrete
		assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))
		angles  = (math.sin(self.rocket.angle), math.cos(self.rocket.angle))


		m_power = 0.0
		if action==2:
			#flipped force
			m_power = 1.0
			main_engine_loc = (self.rocket.position[0] - self.dis_to_main_engine * angles[1], self.rocket.position[1] + self.dis_to_main_engine * angles[0])
			p = self._create_particle(3.5, main_engine_loc[0], main_engine_loc[1], m_power)    # particles are just a decoration, 3.5 is here to make particle speed adequate
			p.ApplyLinearImpulse(           (-angles[1]*MAIN_ENGINE_POWER*m_power,  angles[0]*MAIN_ENGINE_POWER*m_power), main_engine_loc, True)
			self.rocket.ApplyLinearImpulse( (angles[1]*MAIN_ENGINE_POWER*m_power, -angles[0]*MAIN_ENGINE_POWER*m_power), main_engine_loc, True)

		s_power = 0.0
		if action in [1,3]:
			direction = action-2
			s_power = 1
			self.side_engine_loc = (self.rocket.position[0] + direction * self.dis_to_side_engine*angles[0] , self.rocket.position[1] + direction * self.dis_to_side_engine*angles[1])
			p = self._create_particle(0.7, self.side_engine_loc[0], self.side_engine_loc[1], s_power)
			p.ApplyLinearImpulse(           (-angles[0]*direction*SIDE_ENGINE_POWER*s_power,  angles[1]*direction*SIDE_ENGINE_POWER*s_power), self.side_engine_loc, True)
			self.rocket.ApplyLinearImpulse( ( angles[0]*direction*SIDE_ENGINE_POWER*s_power, -angles[1]*direction*SIDE_ENGINE_POWER*s_power), self.side_engine_loc, True)


		#gravity
		for star in self.satellites:
			f = self.attraction(star, self.planet)
			star.ApplyLinearImpulse(f, star.worldCenter, True)
	
		self.world.Step(1.0/FPS, 6*30, 2*30)

		pos = self.rocket.position
		vel = self.rocket.linearVelocity
		state = [
			vel.x*(VIEWPORT_W/SCALE/2)/FPS,
			vel.y*(VIEWPORT_H/SCALE/2)/FPS,
			self.rocket.angle,
			20.0*self.rocket.angularVelocity/FPS
			]
		for star in self.satellites + [self.planet]:
			state.append(pos.x - star.position[0])
			state.append(pos.y - star.position[1])
			if star != self.planet:
				state.append(star.linearVelocity.x*(VIEWPORT_W/SCALE/2)/FPS)
				state.append(star.linearVelocity.y*(VIEWPORT_H/SCALE/2)/FPS)
				state.append(star.angle)


		reward = 0

		reward_temp = self.distance(self.rocket.position, self.planet_pos) - self.planet_radius
		reward = self.prev_reward - reward_temp

		self.prev_reward = reward_temp
		if self.initial_step:
			reward = 0
			self.initial_step = False

		reward -= m_power*0.30  # less fuel spent is better
		reward -= s_power*0.03

		done = False
		if self.game_over or pos.x >= VIEWPORT_W/SCALE or pos.y >= VIEWPORT_H/SCALE \
		or pos.x < 0 or pos.y < 0:
			done   = True
			reward = -1000
		if self.success:
			done   = True
			reward = +1000
		return np.array(state, dtype=np.float32), reward, done, {}


	def render(self, mode='human'):
		from gym.envs.classic_control import rendering
		if self.viewer is None:
			self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
			self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

		for obj in self.particles:
			obj.ttl -= 0.15
			obj.color1 = (max(0.2,0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))
			obj.color2 = (max(0.2,0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))

		self._clean_particles(False)

		self.viewer.draw_polygon(self.universe_poly, color = (self.universe_color + self.torgb(255,255,255)) / 2)

		for star in self.stars:
			self.viewer.draw_polygon(star, color = self.star_color1)
		
		for obj in self.particles + self.drawlist + [self.planet] + self.satellites:
			for f in obj.fixtures:
				trans = f.body.transform
				if type(f.shape) is circleShape:
					t = rendering.Transform(translation=trans*f.shape.pos)
					self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
					self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
				else:
					path = [trans*v for v in f.shape.vertices]
					self.viewer.draw_polygon(path, color=obj.color1)
					path.append(path[0])
					self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def close(self):
		if self.viewer is not None:
			self.viewer.close()
			self.viewer = None
	
	