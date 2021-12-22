import base64
import cv2
from enimda import ENIMDA
from bccaas_engine.upload import UploadFile
import os
from bccaas_api.server import run_server


def check_border(borders):
    if all(borders):
        return False
    else:
        return True


class Check(UploadFile):
    def __init__(self, image_path: str):
        super().__init__(image_path)
        self.write_img(self.image)

    def write_img(self, img):
        cv2.imwrite("img.jpg", img)

    def get_check_name(self) -> str:
        return "Border Check"

    def check_border(self):
        em = ENIMDA(fp="img.jpg", size=200)
        borders = em.scan(fast=True, rows=0.25, threshold=0.2)
        # print(borders)
        if any(borders):
            return True
        else:
            return False

    def get_check_result(self) -> bool:
        return self.check_border()

    def get_remarks(self) -> str:
        if self.check_border():
            return "Border exists"
        else:
            return "No border exist"

    def get_processed_image(self) -> base64:
        retval, buffer = cv2.imencode(".jpg", self.image)
        base64image = base64.b64encode(buffer)
        return base64image


TP = 0
TN = 0
FP = 0
FN = 0

test_dir= 'all_border_dataset_2000'

for i in sorted(os.listdir(test_dir)):
    if i[-15:] == 'Zone.Identifier':
        continue
    image = cv2.imread(f"{test_dir}/{i}")
    res = Check(image)
    if i[:6] == 'border':
        if not res.check_border():
            print(f"{i} is incorrect")
            FN += 1
        else:
            TP += 1
    else:
        if res.check_border():
            print(f"{i} is incorrect")
            FP += 1
        else:
            TN += 1

print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

