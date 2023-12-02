import pygame
import numpy as np
import cv2
import pygame
import numpy as np
from tkinter import Tk, filedialog



CONSTANT = 1
WINDOW_WIDTH = 1160
WINDOW_HEIGHT = 600
BG_COLOR = (0, 0, 255)
BUTTONS_COLOR = (0, 255, 255)

def window_create():
    pygame.init()
    window = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
    pygame.display.set_caption("Redactor")
    
    clock = pygame.time.Clock()
    
    global image_path
    image_path = ""
    global modified_image
    global x
    x = 1
    global is_mody
    is_mody = False
    global text
    text = "1"
    running = True
    global font
    font = pygame.font.Font(None, 32)
    global cv_image
    global resized_image
    resized_image = None
    global is_gray
    is_gray = False
    
    while running:
        
        
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    text = text[:-1]
                elif event.key == pygame.K_RETURN:
                    x = float(text)
                else:
                    text += event.unicode
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if 20 <= mouse_pos[0] <= 120 and 10 <= mouse_pos[1] <= 60:
                    image_path = "*"
                    cv_image = select_image()
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    is_gray = False
                elif 140 <= mouse_pos[0] <= 240 and 10 <= mouse_pos[1] <= 60:
                    modified_image = image_add(cv_image,x)
                    is_mody = True
                    is_gray = False
                elif 260 <= mouse_pos[0] <= 360 and 10 <= mouse_pos[1] <= 60:
                    modified_image = image_negative(cv_image)
                    is_mody = True
                    is_gray = False
                elif 380 <= mouse_pos[0] <= 480 and 10 <= mouse_pos[1] <= 60:
                    modified_image = image_mul(cv_image,x)
                    is_mody = True
                    is_gray = False
                elif 500 <= mouse_pos[0] <= 600 and 10 <= mouse_pos[1] <= 60:
                    modified_image = image_pow(cv_image,x)
                    is_mody = True
                    is_gray = False
                elif 620 <= mouse_pos[0] <= 720 and 10 <= mouse_pos[1] <= 60:
                    modified_image = image_log(cv_image,x)
                    is_mody = True
                    is_gray = False
                elif 740 <= mouse_pos[0] <= 840 and 10 <= mouse_pos[1] <= 60:
                    modified_image = image_linear(cv_image)
                    is_mody = True
                    is_gray = False
                elif 860 <= mouse_pos[0] <= 960 and 10 <= mouse_pos[1] <= 60:
                    modified_image = adaptive_threshold(cv_image, 15, 15)
                    is_mody = True
                    is_gray = True
                elif 980 <= mouse_pos[0] <= 1080 and 10 <= mouse_pos[1] <= 60:
                    modified_image = local_threshold(cv_image,15,15)
                    is_mody = True
                    is_gray = True
        
        window.fill(BG_COLOR)

        
        
        
        if image_path != "":
            resized_image = cv2.resize(cv_image,(460,460))
            resized_image = cv2.rotate(resized_image,cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        global resized_modified_image  
        
        if is_mody:
            resized_modified_image = cv2.resize(modified_image,(460,460))
            resized_modified_image = cv2.rotate(resized_modified_image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        

        
        
        if is_mody:
            global gray_modified_image
            
            gray_modified_image = modified_image
            m_hist = cv2.calcHist([gray_modified_image], [0], None, [256], [0, 256])
            cv2.normalize(m_hist, m_hist, 0, 100, cv2.NORM_MINMAX)
            m_hist_surface = pygame.Surface((500, 100))
            m_hist_surface.fill((255,255,255))
            for i in range(256):
                pygame.draw.line(m_hist_surface, (0, 0, 0), (i, 100), (i, 100 - int(m_hist[i][0])), 1)
            
            m_image_surface = pygame.surfarray.make_surface(resized_modified_image)
            window.blit(m_image_surface, (570, 80))
            #window.blit(m_hist_surface, (570, 80 + resized_modified_image.shape[0]))
        
        if image_path != "":
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, 100, cv2.NORM_MINMAX)
            hist_surface = pygame.Surface((500, 100))
            hist_surface.fill((255,255,255))
            for i in range(256):
                pygame.draw.line(hist_surface, (0, 0, 0), (i, 100), (i, 100 - int(hist[i][0])), 1)
            
            image_surface = pygame.surfarray.make_surface(resized_image)
            window.blit(image_surface, (0, 80))
            #window.blit(hist_surface, (0, 80 + resized_image.shape[0]))
        
        
        draw_buttons(window)
        
        text_surface = font.render(text, True, (0, 0, 0))
        window.blit(text_surface, (1000,540))
        
        pygame.display.flip()

def image_linear(image):
    channels = cv2.split(image)

    eq_channels = []
    for channel in channels:
        eq_channels.append(cv2.equalizeHist(channel))

    result = cv2.merge(eq_channels)
    
    return result

def local_threshold(image, window_size, k):
    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Вычисление локального минимального значения с помощью фильтра Erosion
    min_val = cv2.erode(gray, None, iterations=window_size)

    # Вычисление локального максимального значения с помощью фильтра Dilation
    max_val = cv2.dilate(gray, None, iterations=window_size)

    # Вычисление порогового значения с использованием формулы Берсена
    threshold = 0.5 * (min_val + max_val) - k

    # Применение порогового значения к изображению
    result = np.zeros_like(gray, dtype=np.uint8)
    result[gray > threshold] = 255

    return result

def adaptive_threshold(image, window_size, contrast_threshold):
    
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    gray = cv2.medianBlur(gray,5)
    
    result = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    return result

def image_log(image,x):
    image_float = image.astype(np.float32) / 255.0

    result = np.log(image_float + 1) / np.log(x)

    result *= 255.0
    result = result.astype(np.uint8)
    return result

def image_pow(image,x):
    image_float = image.astype(np.float32) / 255.0
    result = np.power(image_float, x)
    result *= 255.0
    result = result.astype(np.uint8)
    return result

def image_add(image_path,x):
    image = image_path
    r, g, b = cv2.split(image)
    
    
    r_added = cv2.add(r, x)
    g_added = cv2.add(g, x)
    b_added = cv2.add(b, x)
        
    
    added_image = cv2.merge([r_added, g_added, b_added])
    
    return added_image


def image_mul(image,x):
    r, g, b = cv2.split(image)
    
    
    r_mul = cv2.multiply(r, x)
    g_mul = cv2.multiply(g, x)
    b_mul = cv2.multiply(b, x)
        
    
    mul_image = cv2.merge([r_mul, g_mul, b_mul])
    
    return mul_image


def image_negative(image):
    return 255 - image

def draw_buttons(window):
    pygame.draw.rect(window, BUTTONS_COLOR, (20,10, 100, 65))
    font = pygame.font.Font(None, 36)
    text = font.render("Image", True, (0,0,0))
    text_rect = text.get_rect(center=(20 + 100 // 2, 10 + 70 // 2))
    window.blit(text, text_rect)
    
    pygame.draw.rect(window, BUTTONS_COLOR, (140,10, 100, 65))
    font = pygame.font.Font(None, 36)
    text = font.render("+X", True, (0,0,0))
    text_rect = text.get_rect(center=(190 , 10 + 70 // 2))
    window.blit(text, text_rect)
    
    pygame.draw.rect(window, BUTTONS_COLOR, (260,10, 100, 65))
    font = pygame.font.Font(None, 36)
    text = font.render("NEG", True, (0,0,0))
    text_rect = text.get_rect(center=(310, 10 + 70 // 2))
    window.blit(text, text_rect)
    
    pygame.draw.rect(window, BUTTONS_COLOR, (380,10, 100, 65))
    font = pygame.font.Font(None, 36)
    text = font.render("*X", True, (0,0,0))
    text_rect = text.get_rect(center=(430, 10 + 70 // 2))
    window.blit(text, text_rect)
    
    pygame.draw.rect(window, BUTTONS_COLOR, (500,10, 100, 65))
    font = pygame.font.Font(None, 36)
    text = font.render("^X", True, (0,0,0))
    text_rect = text.get_rect(center=(550, 10 + 70 // 2))
    window.blit(text, text_rect)
    
    pygame.draw.rect(window, BUTTONS_COLOR, (620,10, 100, 65))
    font = pygame.font.Font(None, 36)
    text = font.render("Log X", True, (0,0,0))
    text_rect = text.get_rect(center=(670, 10 + 70 // 2))
    window.blit(text, text_rect)
    

    pygame.draw.rect(window, BUTTONS_COLOR, (740,10, 100, 65))
    font = pygame.font.Font(None, 36)
    text = font.render("LC", True, (0,0,0))
    text_rect = text.get_rect(center=(780,  10 + 70 // 2))
    window.blit(text, text_rect)
    
    pygame.draw.rect(window, BUTTONS_COLOR, (860,10, 100, 65))
    font = pygame.font.Font(None, 36)
    text = font.render("A T", True, (0,0,0))
    text_rect = text.get_rect(center=(900, 40))
    window.blit(text, text_rect)
    
    pygame.draw.rect(window, BUTTONS_COLOR, (980,10, 100, 65))
    font = pygame.font.Font(None, 36)
    text = font.render("L T", True, (0,0,0))
    text_rect = text.get_rect(center=(1020, 40))
    window.blit(text, text_rect)

def select_image():
    root = Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title="Выберите картинку", filetypes=[("Изображения", "*.png;*.jpg;*.jpeg")])
    image = cv2.imread(image_path)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        return image
    elif len(image.shape) == 3 and image.shape[2] == 3:
        return image


if __name__ == "__main__":
    window_create()