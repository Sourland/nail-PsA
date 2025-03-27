# Nail Psoriasis Detection
A repository containing the source code of my Electrical and Computer Engineering Degree Diploma Thesis

Psoriatic arthritis occurs in 30-40% of patients suffering from cutaneous psoriasis and is a chronic inflammatory disease of the joints. It manifests itself with symptoms such as pain, stiffness, and swelling of the joints, significantly affecting patients’ quality of life. The pathophysiological mechanisms that drive and determine the course of the disease are not fully understood, and its symptoms are similar to other arthritis. Consequently, the early diagnosis and treatment of psoriatic arthritis still presents several challenges today. Psoriasis in the nails and psoriatic arthritis can have many symptoms, such as swollen joints, pits in the nail, hyperkeratosis under the nail or onycholysis. While the diagnosis of psoriasis in the nails can be made visually by a dermatologist, usually the diagnosis of psoriatic arthritis requires the use of x-rays to study the bones in the joints, which makes the process very difficult. As the number of patients with psoriasis and psoriatic arthritis increase, it is of paramount importance to have an easy, quick but reliable diagnosis of the disease.

In the development of such a solution, technology and in particular artificial intelligence can be of great help. In particular, it is possible to extract the width of a joint from an RGB photograph for the purpose of evaluating it. When joints swell, essentially the bone swells, and the finger is thicker in the area of the joint. A person who does not have psoriatic arthritis will have a thinner joint than a patient, which can be used as a differential factor in an initial diagnosis. On the other hand, through neural networks in classification problems, it is possible to evaluate photographs of fingernails for any symptoms.

In this thesis, a method for psoriasis evaluation using RGB images is developed. A method of extracting biomarkers from the hand in the form of joint width is described, which uses image segmentation, detection of joints by computer vision and biomarker extraction, and the difference between healthy joints and swollen joints is analyzed statistically. In addition, after experiments, a DenseNet121 neural network is developed and trained to classify symptoms of psoriasis in the nail by transfer learning method with an accuracy of 86.77% and a class activation map is used, using GradCAM method, which offer explanation to the classification options of DenseNet121 based on the features extracted, which provides reliability to the system.

You can the the report [here](https://ikee.lib.auth.gr/record/354824/files/SOURLANTZIS_DIMITRIOS.pdf)

BUILD INSTRUCTIONS COMING SOON
