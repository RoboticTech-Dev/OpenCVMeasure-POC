import cv2


class Frechau:
    def __init__(self):
        self.kernel = cv2.imread('kernel.png', flags=cv2.IMREAD_GRAYSCALE)

    def find_coordinates(self, image):
        # Match value corresponds to sum difference squared
        match_value = 1e9
        y_goal = 0

        # Iterate over y direction to get best match
        if image.shape[0] > self.kernel.shape[0]:
            if image.shape[1] > self.kernel.shape[1]:

                for i in range(0, image.shape[1] - self.kernel.shape[1] - 2):
                    image_temp = image[:, i:i+image.shape[1]-1]
                    value = sum(sum( (image_temp - self.kernel)**2 ))
                    if value < match_value:
                        match_value, y_goal = value, i

            else:
                print("Altura da imagem menor que a altura do proprio kernel, ajustar.")
        else:
             print("Largura da imagem medida e menor que a do Kernel, ajustar.")

        # Return best coordinate plus some extra pixels
        return 0 if y_goal-5 < 0 else y_goal