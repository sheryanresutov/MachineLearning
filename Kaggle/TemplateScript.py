from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, enet_path, Lasso
from sklearn.svm import SVC

trainFile = "./Data/train_data.csv"
testFile = "./Data/test_x_data.csv"
with open(trainFile,mode='rb') as file:
            train = pd.DataFrame(pd.read_csv(file))

X = train.values.tolist()
X = np.asarray(X,dtype=np.double)
b = [i[561] for i in X]
b = np.asarray(b,dtype=np.double)
cv = cross_validation.KFold(len(X), n_folds=5)
rf_count=0
svm_count=0

feat_imp = [41 , 53 , 51 , 560 , 54 , 57 , 559 , 42 , 50 , 58 , 43 , 52 , 561 , 55 , 59 , 509 , 4 , 269 , 272 , 10 , 75 , 17 , 504 , 282 , 182 , 74 , 451 , 311 , 38 , 266 , 394 , 361 , 64 , 56 , 202 , 169 , 348 , 390 , 103 , 76 , 354 , 216 , 441 , 96 , 70 , 275 , 40 , 253 , 89 , 303 , 206 , 351 , 7 , 66 , 382 , 105 , 505 , 71 , 215 , 315 , 258 , 77 , 72 , 203 , 469 , 133 , 87 , 503 , 176 , 227 , 424 , 410 , 235 , 398 , 44 , 127 , 67 , 304 , 506 , 124 , 349 , 352 , 223 , 356 , 383 , 201 , 97 , 365 , 179 , 427 , 210 , 68 , 180 , 140 , 73 , 130 , 217 , 516 , 219 , 430 , 440 , 177 , 281 , 183 , 446 , 85 , 360 , 405 , 78 , 363 , 63 , 204 , 404 , 234 , 160 , 473 , 445 , 47 , 229 , 429 , 396 , 459 , 181 , 88 , 297 , 462 , 185 , 167 , 260 , 273 , 143 , 367 , 84 , 69 , 461 , 20 , 220 , 422 , 296 , 214 , 166 , 104 , 207 , 23 , 233 , 164 , 345 , 288 , 39 , 137 , 442 , 362 , 208 , 173 , 489 , 510 , 259 , 48 , 519 , 452 , 101 , 186 , 228 , 508 , 487 , 232 , 412 , 392 , 513 , 128 , 298 , 222 , 100 , 125 , 511 , 60 , 80 , 294 , 209 , 141 , 98 , 285 , 45 , 435 , 433 , 118 , 448 , 136 , 317 , 245 , 425 , 449 , 439 , 86 , 93 , 346 , 387 , 139 , 142 , 129 , 418 , 211 , 450 , 517 , 126 , 497 , 538 , 447 , 395 , 102 , 369 , 221 , 61 , 158 , 460 , 471 , 146 , 370 , 224 , 1 , 536 , 391 , 170 , 90 , 121 , 386 , 175 , 406 , 305 , 431 , 432 , 295 , 13 , 291 , 523 , 286 , 408 , 18 , 5 , 299 , 388 , 319 , 331 , 46 , 501 , 138 , 49 , 79 , 92 , 172 , 428 , 159 , 458 , 135 , 16 , 539 , 19 , 475 , 483 , 123 , 545 , 522 , 434 , 300 , 426 , 333 , 340 , 284 , 397 , 524 , 457 , 512 , 353 , 276 , 465 , 199 , 329 , 145 , 11 , 240 , 292 , 464 , 188 , 8 , 339 , 99 , 372 , 491 , 470 , 24 , 466 , 347 , 3 , 6 , 21 , 490 , 302 , 350 , 132 , 9 , 419 , 454 , 325 , 318 , 484 , 65 , 271 , 443 , 463 , 198 , 544 , 62 , 200 , 525 , 518 , 244 , 254 , 343 , 558 , 472 , 14 , 246 , 26 , 178 , 313 , 293 , 326 , 498 , 131 , 12 , 467 , 364 , 557 , 168 , 444 , 283 , 2 , 270 , 301 , 456 , 474 , 241 , 134 , 521 , 150 , 530 , 165 , 548 , 476 , 157 , 15 , 91 , 499 , 373 , 355 , 25 , 528 , 274 , 488 , 540 , 455 , 534 , 453 , 529 , 255 , 122 , 22 , 515 , 502 , 411 , 120 , 161 , 261 , 119 , 95 , 287 , 543 , 289 , 535 , 531 , 492 , 423 , 514 , 256 , 375 , 478 , 527 , 154 , 542 , 477 , 413 , 247 , 532 , 267 , 144 , 316 , 553 , 312 , 197 , 277 , 205 , 366 , 554 , 344 , 248 , 547 , 184 , 243 , 541 , 109 , 242 , 263 , 384 , 307 , 334 , 190 , 500 , 332 , 171 , 174 , 290 , 257 , 371 , 549 , 218 , 480 , 249 , 468 , 151 , 308 , 163 , 149 , 401 , 37 , 330 , 106 , 94 , 409 , 385 , 378 , 29 , 306 , 116 , 231 , 193 , 327 , 485 , 320 , 268 , 374 , 262 , 230 , 321 , 496 , 322 , 526 , 155 , 341 , 314 , 551 , 82 , 377 , 187 , 148 , 379 , 414 , 250 , 495 , 189 , 493 , 153 , 415 , 108 , 156 , 400 , 479 , 115 , 494 , 265 , 368 , 407 , 213 , 252 , 376 , 537 , 226 , 33 , 83 , 194 , 117 , 239 , 555 , 30 , 34 , 225 , 380 , 438 , 264 , 486 , 550 , 236 , 310 , 196 , 399 , 162 , 191 , 113 , 195 , 110 , 328 , 420 , 402 , 552 , 237 , 556 , 192 , 212 , 393 , 114 , 546 , 81 , 436 , 381 , 309 , 147 , 533 , 323 , 279 , 111 , 335 , 482 , 416 , 152 , 251 , 481 , 112 , 337 , 107 , 31 , 32 , 357 , 28 , 437 , 336 , 280 , 338 , 238 , 421 , 324 , 342 , 35 , 507 , 36 , 359 , 278 , 27 , 417 , 358 , 389 , 403 , 520 ]
feat_imp_subset = feat_imp[560:]
    
feature_importance=[]

runRandomForest = raw_input("Do you want to run Random Forest?")

runSVM = raw_input("Do you want to run SVM?")

runOnTestData = raw_input("Do you want to run on test Data?")

for traincv, testcv in cv:
    train_set=X[traincv]
    test_set = np.delete(X[testcv],561,1)
    label_set= b[testcv]
    
    train_data = np.delete(train_set,561,1)
    target_data = [data[561] for data in train_set]
        
    #Random Forest
    if(runRandomForest == 'y'):
        rf = RandomForestClassifier(n_estimators=200, max_features='log2', bootstrap=False)
        rf.fit(train_data, target_data)
        rf_predicted = rf.predict(test_set)
        for i in range(len(label_set)):
            if(label_set[i] != rf_predicted[i]):
                rf_count=rf_count+1

    #Support Vector Machines
    if(runSVM == 'y'):
        clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
        gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
        random_state=None, shrinking=True, tol=0.001, verbose=False)
        clf.fit(train_data, target_data)
        svm_predicted = clf.predict(test_set)
        for i in range(len(label_set)):
            if(label_set[i] != svm_predicted[i]):
                svm_count=svm_count+1

if(runOnTestData == 'y'):
    with open(testFile,mode='rb') as file:
        test = pd.DataFrame(pd.read_csv(file))
    X_test=test.values.tolist()
    X_test=np.asarray(X_test, dtype=np.double)
    rf_test_predicted = rf.predict(X_test)
    np.savetxt("test_y_data.csv",rf_test_predicted)


#np.savetxt("feature_importances.csv", np.asarray(feature_importance,dtype=np.double))
if(runRandomForest == 'y'):
    print("Random Forest accuracy: ")
    print(1-rf_count/7767.000)

if(runSVM == 'y'):
    print("SVM accuracy: ")
    print(1-svm_count/7767.000)
