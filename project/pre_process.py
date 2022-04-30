import cv2
import os
import numpy as np


class Extractor:
    def __init__(self, path):
        self.image = cv2.imread(path, 0)
        self.originalimage = np.copy(self.image)
        self.extractgrid = None

    def image_preprocess(self):
        img = self.image
        img = cv2.GaussianBlur(img, (11, 11), 0)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 5, 2)
        img = cv2.bitwise_not(img)
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        # img = cv2.dilate(img, kernel)
        maxi = -1
        maxpt = None
        value = 10
        height, width = np.shape(img)
        for h in range(height):
            row = img[h]
            for w in range(width):
                if row[w] >= 128:
                    area = cv2.floodFill(img, None, (w, h), 64)[0]
                    if value > 0:
                        value -= 1
                    if area > maxi:
                        maxpt = (w, h)
                        maxi = area
        cv2.floodFill(img, None, maxpt, (255, 255, 255))
        for h in range(height):
            row = img[h]
            for w in range(width):
                if row[w] == 64 and w != maxpt[0] and h != maxpt[1]:
                    cv2.floodFill(img, None, (w, h), 0)
        kernel1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        img = cv2.erode(img, kernel1)

        def drawLine(line, img):
            h, w = np.shape(img)
            if line[0][1] != 0:
                m = -1 / np.tan(line[0][1])
                c = line[0][0] / np.sin(line[0][1])
                cv2.line(img, (0, int(c)), (w, int(m * w + c)), 255)
            else:
                cv2.line(img, (line[0][0], 0), (line[0][0], h), 255)
            return img

        lines = cv2.HoughLines(img, 1, np.pi / 180, 200)
        temp_img = np.copy(img)
        # for i in range(len(lines)):
        #    temp_img = drawLine(lines[i], temp_img)

        def mergeLines(lines, img):
            h, w = np.shape(img)
            for line in lines:
                if line[0][0] is None and line[0][1] is None:
                    continue
                p1 = line[0][0]
                theta1 = line[0][1]
                pt1current = [None, None]
                pt2current = [None, None]
                if np.pi * 45 / 180 < theta1 < np.pi * 135 / 180:
                    pt1current[0] = 0
                    pt1current[1] = p1 / np.sin(theta1)
                    pt2current[0] = w
                    pt2current[1] = -1*pt2current[0] / np.tan(theta1) * p1 / np.sin(theta1)
                else:
                    pt1current[1] = 0
                    pt1current[0] = p1 / np.cos(theta1)
                    pt2current[1] = h
                    pt2current[0] = -1*pt2current[1] / np.tan(theta1) * p1 / np.cos(theta1)
                for pos in lines:
                    if pos[0].all() == line[0].all():
                        continue
                    if abs(pos[0][0] - line[0][0]) < 20 and abs(pos[0][1] - line[0][1]) < np.pi * 10 / 180:
                        p = pos[0][0]
                        theta = pos[0][1]
                        pt1 = [None, None]
                        pt2 = [None, None]
                        if np.pi * 45 / 180 < theta1 < np.pi * 135 / 180:
                            pt1[0] = 0
                            pt1[1] = p1 / np.sin(theta1)
                            pt2[0] = w
                            pt2[1] = -pt2[0] / np.tan(theta1) * p1 / np.sin(theta1)
                        else:
                            pt1[1] = 0
                            pt1[0] = p1 / np.cos(theta1)
                            pt2[1] = h
                            pt2[0] = -pt2current[0] / np.tan(theta1) * p1 / np.cos(theta1)
                        if ((pt1[0] - pt1current[0]) ** 2 + (pt1[1] - pt1current[1]) ** 2 < 64 ** 2) and (
                                (pt2[0] - pt2current[0]) ** 2 + (pt2[1] - pt2current[1]) ** 2 < 64 ** 2):
                            line[0][0] = (line[0][0] + pos[0][0]) / 2
                            line[0][1] = (line[0][1] + pos[0][1]) / 2
                            pos[0][0] = None
                            pos[0][1] = None
            lines = list(filter(lambda a: a[0][0] is not None and a[0][1] is not None, lines))
            return lines

        lines = mergeLines(lines, img)
        topedge = [[1000, 1000]]
        bottomedge = [[-1000, -1000]]
        leftedge = [[1000, 1000]]
        leftxintercept = 100000
        rightedge = [[-1000, -1000]]
        rightxintercept = 0
        for i in range(len(lines)):
            current = lines[i][0]
            p = current[0]
            theta = current[1]
            xIntercept = p / np.cos(theta)

            # If the line is nearly vertical
            if np.pi * 80 / 180 < theta < np.pi * 100 / 180:
                if p < topedge[0][0]:
                    topedge[0] = current[:]
                if p > bottomedge[0][0]:
                    bottomedge[0] = current[:]
            if theta < np.pi * 10 / 180 or theta > np.pi * 170 / 180:
                if xIntercept > rightxintercept:
                    rightedge[0] = current[:]
                    rightxintercept = xIntercept
                elif xIntercept <= leftxintercept:
                    leftedge[0] = current[:]
                    leftxintercept = xIntercept

        tmpimg = np.copy(img)
        tmppp = np.copy(self.originalimage)
        tmppp = drawLine(leftedge, tmppp)
        tmppp = drawLine(rightedge, tmppp)
        tmppp = drawLine(topedge, tmppp)
        tmppp = drawLine(bottomedge, tmppp)

        tmpimg = drawLine(leftedge, tmpimg)
        tmpimg = drawLine(rightedge, tmpimg)
        tmpimg = drawLine(topedge, tmpimg)
        tmpimg = drawLine(bottomedge, tmpimg)
        leftedge = leftedge[0]
        rightedge = rightedge[0]
        bottomedge = bottomedge[0]
        topedge = topedge[0]

        left1 = [None, None]
        left2 = [None, None]
        right1 = [None, None]
        right2 = [None, None]
        top1 = [None, None]
        top2 = [None, None]
        bottom1 = [None, None]
        bottom2 = [None, None]

        if leftedge[1] != 0:
            left1[0] = 0
            left1[1] = leftedge[0] / np.sin(leftedge[1])
            left2[0] = width
            left2[1] = -left2[0] / np.tan(leftedge[1]) + left1[1]
        else:
            left1[1] = 0
            left1[0] = leftedge[0] / np.cos(leftedge[1])
            left2[1] = height
            left2[0] = left1[0] - height * np.tan(leftedge[1])

        if rightedge[1] != 0:
            right1[0] = 0
            right1[1] = rightedge[0] / np.sin(rightedge[1])
            right2[0] = width
            right2[1] = -right2[0] / np.tan(rightedge[1]) + right1[1]
        else:
            right1[1] = 0
            right1[0] = rightedge[0] / np.cos(rightedge[1])
            right2[1] = height
            right2[0] = right1[0] - height * np.tan(rightedge[1])

        bottom1[0] = 0
        bottom1[1] = bottomedge[0] / np.sin(bottomedge[1])

        bottom2[0] = width
        bottom2[1] = -bottom2[0] / np.tan(bottomedge[1]) + bottom1[1]

        top1[0] = 0
        top1[1] = topedge[0] / np.sin(topedge[1])
        top2[0] = width
        top2[1] = -top2[0] / np.tan(topedge[1]) + top1[1]

        leftA = left2[1] - left1[1]
        leftB = left1[0] - left2[0]
        leftC = leftA * left1[0] + leftB * left1[1]

        rightA = right2[1] - right1[1]
        rightB = right1[0] - right2[0]
        rightC = rightA * right1[0] + rightB * right1[1]

        topA = top2[1] - top1[1]
        topB = top1[0] - top2[0]
        topC = topA * top1[0] + topB * top1[1]

        bottomA = bottom2[1] - bottom1[1]
        bottomB = bottom1[0] - bottom2[0]
        bottomC = bottomA * bottom1[0] + bottomB * bottom1[1]
        detTopLeft = leftA * topB - leftB * topA
        ptTopLeft = ((topB * leftC - leftB * topC) / detTopLeft, (leftA * topC - topA * leftC) / detTopLeft)
        detTopRight = rightA * topB - rightB * topA
        ptTopRight = ((topB * rightC - rightB * topC) / detTopRight, (rightA * topC - topA * rightC) / detTopRight)
        detBottomRight = rightA * bottomB - rightB * bottomA
        ptBottomRight = (
            (bottomB * rightC - rightB * bottomC) / detBottomRight, (rightA * bottomC - bottomA * rightC) / detBottomRight)
        detBottomLeft = leftA * bottomB - leftB * bottomA
        ptBottomLeft = ((bottomB * leftC - leftB * bottomC) / detBottomLeft,
                        (leftA * bottomC - bottomA * leftC) / detBottomLeft)
        cv2.circle(tmppp, (int(ptTopLeft[0]), int(ptTopLeft[1])), 5, 0, -1)
        cv2.circle(tmppp, (int(ptTopRight[0]), int(ptTopRight[1])), 5, 0, -1)
        cv2.circle(tmppp, (int(ptBottomLeft[0]), int(ptBottomLeft[1])), 5, 0, -1)
        cv2.circle(tmppp, (int(ptBottomRight[0]), int(ptBottomRight[1])), 5, 0, -1)
        leftedgelensq = (ptBottomLeft[0] - ptTopLeft[0]) ** 2 + (ptBottomLeft[1] - ptTopLeft[1]) ** 2
        rightedgelensq = (ptBottomRight[0] - ptTopRight[0]) ** 2 + (ptBottomRight[1] - ptTopRight[1]) ** 2
        topedgelensq = (ptTopRight[0] - ptTopLeft[0]) ** 2 + (ptTopLeft[1] - ptTopRight[1]) ** 2
        bottomedgelensq = (ptBottomRight[0] - ptBottomLeft[0]) ** 2 + (ptBottomLeft[1] - ptBottomRight[1]) ** 2
        maxlength = int(max(leftedgelensq, rightedgelensq, bottomedgelensq, topedgelensq) ** 0.5)
        src = [(0, 0)] * 4
        dst = [(0, 0)] * 4
        src[0] = ptTopLeft[:]
        dst[0] = (0, 0)
        src[1] = ptTopRight[:]
        dst[1] = (maxlength - 1, 0)
        src[2] = ptBottomRight[:]
        dst[2] = (maxlength - 1, maxlength - 1)
        src[3] = ptBottomLeft[:]
        dst[3] = (0, maxlength - 1)
        src = np.array(src).astype(np.float32)
        dst = np.array(dst).astype(np.float32)
        self.extractedgrid = cv2.warpPerspective(self.originalimage, cv2.getPerspectiveTransform(src, dst),
                                                 (maxlength, maxlength))
        self.extractedgrid = cv2.resize(self.extractedgrid, (252, 252))
        grid = np.copy(self.extractedgrid)
        edge = np.shape(grid)[0]
        celledge = edge // 9
        grid = cv2.bitwise_not(cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1))
        tempgrid = []
        for i in range(celledge,edge+1, celledge):
            for j in range(celledge, edge+1, celledge):
                rows = grid[i-celledge:i]
                tempgrid.append([rows[k][j-celledge:j] for k in range(len(rows))])
        finalgrid = []
        for i in range(0, len(tempgrid)-8, 9):
            finalgrid.append(tempgrid[i:i+9])
        for i in range(9):
            for j in range(9):
                finalgrid[i][j] = np.array(finalgrid[i][j])
        try:
            for i in range(9):
                for j in range(9):
                    os.remove("cell"+str(i)+str(j)+".jpg")
        except:
            pass
        for i in range(9):
            for j in range(9):
                cv2.imwrite(str("cell"+str(i)+str(j)+".jpg"),finalgrid[i][j])
        self.image = img
        return finalgrid
