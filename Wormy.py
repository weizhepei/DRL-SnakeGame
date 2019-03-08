
from pygame.locals import *
import random, pygame, sys,math
import os
#os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

GAME = 'wormy'
FPS = 30
TOTAL_LIFES = 200000
TRAIN_LIFES = 150000
FINAL_ALPHA = 0.5
INITIAL_ALPHA = 0.7
FINAL_EPSILON = 0
INITIAL_EPSILON = 0.5
WINDOWWIDTH = 240
WINDOWHEIGHT = 240
CELLSIZE = 20
assert WINDOWWIDTH % CELLSIZE == 0, "Window width must be a multiple of cell size."
assert WINDOWHEIGHT % CELLSIZE == 0, "Window height must be a multiple of cell size."
CELLWIDTH = int(WINDOWWIDTH / CELLSIZE)
CELLHEIGHT = int(WINDOWHEIGHT / CELLSIZE)

WHITE     = (255, 255, 255)
BLACK     = (  0,   0,   0)
RED       = (255,   0,   0)
GREEN     = (  0, 255,   0)
DARKGREEN = (  0, 155,   0)
DARKGRAY  = ( 100,  100,  100)
BGCOLOR = BLACK

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

HEAD = 0
pygame.init()
pygame.display.init()
FPSCLOCK = pygame.time.Clock()
DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
pygame.display.set_caption('Snake Game')


class GameState:
	Total_Score = 0
	Lifes = 0
	Epsilon = INITIAL_EPSILON
	alpha = INITIAL_ALPHA
	def getEpsilon(self):
		return self.Epsilon
	def getAlpha(self):
		return self.alpha
	def getScore(self):
		return self.score
	def getLifes(self):
		return self.Lifes
	def getAlive(self):
		return self.alive_step
	def getLength(self):
		return self.size
	def getDirection(self):
		if(self.direction == LEFT):
			return 0
		elif(self.direction == UP):
			return 1
		elif(self.direction == DOWN):
			return  2
		else: return 3
	def __init__(self):
		if(self.getLifes() > 0):
			self.Epsilon -=  (INITIAL_EPSILON - FINAL_EPSILON)/ (TRAIN_LIFES)
			self.alpha -= (INITIAL_ALPHA - FINAL_ALPHA) / (TRAIN_LIFES/2)
			if(self.Epsilon < FINAL_EPSILON):
				self.Epsilon = FINAL_EPSILON
			if(self.alpha < FINAL_ALPHA):
				self.alpha = FINAL_ALPHA
		self.Lifes += 1
		self.time_out = 0
		self.hang_out = 0
		self.score = 0
		self.alive_step = 0
		self.startx = random.randint(2, CELLWIDTH - 2)
		self.starty = random.randint(0, CELLHEIGHT - 1)
		self.wormCoords = [{'x': self.startx, 'y': self.starty},
			{'x': self.startx - 1, 'y': self.starty},
			{'x': self.startx - 2, 'y': self.starty}]
		self.size = 3
		self.direction = RIGHT
		self.apple = getRandomLocation(self.wormCoords)
		DISPLAYSURF.fill(BGCOLOR)
		drawGrid()
		drawApple(self.apple)
		drawWorm(self.wormCoords)
		pygame.display.update()
		
	def frame_step(self,input_actions):
		self.time_out += 1
		self.alive_step += 1
		DISPLAYSURF.fill(BGCOLOR)
		drawGrid()
		drawWorm(self.wormCoords)
		drawApple(self.apple)
		pygame.display.update()
		pygame.event.pump()
		reward = 0
		terminal = False
		flag_timeout = False
		if sum(input_actions) > 1:
			raise ValueError('Multiple input actions!')
		elif sum(input_actions) == 0:
			s_do_nothing = pygame.surfarray.array3d(pygame.display.get_surface())
			return s_do_nothing, 0, False, False
		
		if input_actions[0] == 1 and self.direction != RIGHT:
			self.direction = LEFT
		elif input_actions[1] ==1 and self.direction != DOWN:
			self.direction = UP
		elif input_actions[2] ==1 and self.direction != UP:
			self.direction = DOWN
		elif input_actions[3] ==1 and self.direction != LEFT:
			self.direction = RIGHT
			
		if self.direction == UP:
			newHead = {'x': self.wormCoords[HEAD]['x'], 'y': self.wormCoords[HEAD]['y'] - 1}
		elif self.direction == DOWN:
			newHead = {'x': self.wormCoords[HEAD]['x'], 'y': self.wormCoords[HEAD]['y'] + 1}
		elif self.direction == LEFT:
			newHead = {'x': self.wormCoords[HEAD]['x'] - 1, 'y': self.wormCoords[HEAD]['y']}
		elif self.direction == RIGHT:
			newHead = {'x': self.wormCoords[HEAD]['x'] + 1, 'y': self.wormCoords[HEAD]['y']}
				
		if newHead['x'] == self.apple['x'] and newHead['y'] == self.apple['y']:
			self.score += 1
			self.size += 1
			if (self.hang_out == 0):
				reward = 1
			self.size += 1
			self.time_out = 0
			if (self.size > 10):
				self.hang_out = math.ceil(0.4 * self.size) + 2
			else:
				self.hang_out = 6
		else:
			del self.wormCoords[-1]
			DISPLAYSURF.fill(BGCOLOR)
			drawGrid()
			drawWorm(self.wormCoords)
			pygame.display.update()
			
		if newHead['x'] == -1 or newHead['x'] == CELLWIDTH or newHead['y'] == -1 or newHead['y'] == CELLHEIGHT:
			terminal = True
			self.__init__()
			reward = -1
		for wormBody in self.wormCoords[1:]:
			if wormBody['x'] == self.wormCoords[HEAD]['x'] and wormBody['y'] == self.wormCoords[HEAD]['y']:
				terminal = True
				self.__init__()
				reward = -1
		
		if (self.wormCoords[HEAD]['x'] == self.apple['x'] and self.wormCoords[HEAD]['y'] == self.apple['y']):
			self.apple = getRandomLocation(self.wormCoords)
		
		if (not terminal):
			dis_x_old = self.wormCoords[HEAD]['x'] - self.apple['x']
			dis_y_old = self.wormCoords[HEAD]['y'] - self.apple['y']
			dis_x_new = newHead['x'] - self.apple['x']
			dis_y_new = newHead['y'] - self.apple['y']
			dis_old = math.sqrt(dis_x_old * dis_x_old + dis_y_old * dis_y_old)
			dis_new = math.sqrt(dis_x_new * dis_x_new + dis_y_new * dis_y_new)
			if(self.hang_out == 0 and dis_old * dis_new != 0):
				reward += math.log((self.size + dis_old)/(self.size + dis_new)) / math.log(self.size)

			self.wormCoords.insert(0, newHead)
			drawWorm(self.wormCoords)
			drawApple(self.apple)
			pygame.display.update()

			if(self.time_out >= math.ceil(self.size * 0.7 + 10)):
				reward -= 0.5/self.size
				flag_timeout = True
				self.time_out = 0
			if(self.hang_out != 0):
				self.hang_out -= 1
				
		if(reward > 1):
			reward = 1
		elif(reward < -1):
			reward = -1
			
		image_data = pygame.surfarray.array3d(pygame.display.get_surface())
		FPSCLOCK.tick(FPS)
		return image_data,reward,terminal,flag_timeout

def getRandomLocation(wormCoords):
	while True:
		temp =  {'x': random.randint(0, CELLWIDTH - 1), 'y': random.randint(0, CELLHEIGHT - 1)}
		flag = True
		for wormBody in wormCoords[0:]:
			if wormBody['x'] == temp['x'] and wormBody['y'] == temp['y']:
				flag = False
		if flag:
			break
	if flag:
		return temp

def drawWorm(wormCoords):
	for coord in wormCoords:
		x = coord['x'] * CELLSIZE
		y = coord['y'] * CELLSIZE
		wormSegmentRect = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
		pygame.draw.rect(DISPLAYSURF, DARKGREEN, wormSegmentRect)
		wormInnerSegmentRect = pygame.Rect(x + 4, y + 4, CELLSIZE - 8, CELLSIZE - 8)
		pygame.draw.rect(DISPLAYSURF, GREEN, wormInnerSegmentRect)


def drawApple(coord):
	x = coord['x'] * CELLSIZE
	y = coord['y'] * CELLSIZE
	appleRect = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
	pygame.draw.rect(DISPLAYSURF, RED, appleRect)


def drawGrid():
	for x in range(0, WINDOWWIDTH, CELLSIZE):
		pygame.draw.line(DISPLAYSURF, DARKGRAY, (x, 0), (x, WINDOWHEIGHT))
	for y in range(0, WINDOWHEIGHT, CELLSIZE):
		pygame.draw.line(DISPLAYSURF,DARKGRAY,(0,y),(WINDOWWIDTH,y))


#def drawScore(score):
#    scoreSurf = BASICFONT.render('Score: %s' % (score), True, WHITE)
#    scoreRect = scoreSurf.get_rect()
#    scoreRect.topleft = (WINDOWWIDTH - 120, 10)
#    DISPLAYSURF.blit(scoreSurf, scoreRect)




