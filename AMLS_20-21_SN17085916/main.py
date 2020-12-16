import os
from A1.gender import Gender
from A2.emotion import Emotion
from B1.face_shape import Face_shape
from B2.eye_color import Eye_color


def main():
    gender = Gender()
    emotion = Emotion()
    face_shape = Face_shape()
    eye_color = Eye_color()
    # 测试Gender
    if os.path.exists('A1/model/model.pth'):
        gender.test()
    else:
        gender.train()
        gender.test()
    # 测试Emotion
    if os.path.exists('A2/model/model.pth'):
        emotion.test()
    else:
        emotion.train()
        emotion.test()
    # 测试Face_shape
    if os.path.exists('B1/model/model.pth'):
        face_shape.test()
    else:
        face_shape.train()
        face_shape.test()
    # 测试Eye_color
    if os.path.exists('B2/model/model.pth'):
        eye_color.test()
    else:
        eye_color.train()
        eye_color.test()


if __name__ == "__main__":

    main()
