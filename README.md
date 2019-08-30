# WasteClassification

Prerequisites:
* To use fast.ai library on anaconda, install pytorch

Note:
Due to exceeding GitHub's file size limit of 100.00 MB, the exported keras model h5 file is here https://drive.google.com/file/d/1PBldPjErDYTofAozvpxnFdMRnNatz29G/view?usp=sharing

Recourses:

* Get data from https://github.com/garythung/trashnet/blob/master/data/dataset-resized.zip
* Data preprocessing part partially depends on Vishcam's waste classifier project https://github.com/KhazanahAmericasInc/waste-classifier
* Model training (PyTorch version) part partially depends on Collin Ching's waste sorter project https://github.com/collindching/Waste-Sorter



# Outlines

-   Waste Classification

    -   android application with google ML kit

        -   set up environment

            -   update the android studio from 2.3.2 to 3.4.1 so that it
                will have the gradle version \> 4.0, which is required
                by ML Kit

                -   The current gradel it support is version 5.1.1.

            -   Downloading system image (oreo 26)

            -   Run into this issue

                -   <https://stackoverflow.com/questions/51782548/androidxappcompat-iart-error-android-view-viewonunhandledkeyeventlistener/52136900>

                    -   Initially used android API level 26 because I
                        think it fit to more devices.

                        -   Now switching to API level 28 to avoid the
                            init-issue.

        -   main features

            -   camera interface

                -   using android Camera API

            -   orientation

                -   force portrait orientation, disable landscape
                    orientation

            -   connected to firebase

                -   upload image to firebase storage

                    -   based on different waste type

            -   firebase storage organization

                -   image naming

            -   predicting waste type

                -   calling automl model

            -   if "unable to predict"

                -   send the image and prediction to real\_time database
                    in firebase

            -   ask for validation from users

                -   if the predicted type is correct

                    -   do nothing but thanks

                -   otherwise

                    -   ask user to select the correct type and upload
                        to firebase storage

            -   welcome page

                -   start classification

            -   thanking page

                -   continue classification

        -   evaluation

            -   I was testing the application on my phone, some of the
                images get prediction result very fast, but some images
                took forever to get the prediction.

                -   have been searching for workaround but have no
                    solution

                    -   but later on, tried several times later,
                        predicting speed is get better and better.

        -   two models

            -   model trained by AutoML

                -   The AutoML is a new feature on google ML Kit, so
                    there is not much project using it yet. I will use
                    my own model on google ML Kit instead of the one
                    trained by AutoML tomorrow.

            -   customized model trained by Clair

                -   after adding customized model

                    -   publish model

                        -   rejected, due to model size limitation
