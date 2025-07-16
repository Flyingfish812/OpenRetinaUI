# Open Retina UI

Informal (personal) UI building for Open retina repository.

Use this UI after you installed [Open retina](https://github.com/open-retina/open-retina)

The project is using gradio UI, please install it in your environment first.

We do not offer dataset for confidential reasons. But you can adapt the code to your proper dataset.

## Version 0.1.0

- 2025/06/09

Initial commit

## Version 0.1.1

- 2025/06/10

Visualizer page added

## Version 0.1.2

- 2025/06/11

Add optimizer and scheduler settings, fixed data normalization methods.

## Version 0.1.3

- 2025/06/12

3D model fit, test on open retina built-in models.

## Version 0.1.4

- 2025/06/16

LSTA plotting added, personalized activation functions enabled.

## Version 0.2.0

- 2025/06/19

The main functions of this UI application is finished, You can use this UI along with open retina to test your own model:

- Data Preprocessing: `images_train`, `responses_train`, `images_val`, `responses_val`, `image_test`, `response_test`.
- Dataloader Saving and Loading
- Model Building and Settings Saving
- Personalized Training Settings
- Visualization on Model Metrics

## Version 0.2.1

- 2025/06/20

Small bug fixed. Corrected R2 plot enabled.

## Version 0.2.2

- 2025/06/20

LNLN model enabled.

## Version 0.2.3

- 2025/06/24

LSTA plot optimized.

## Version 0.3.0

- 2025/07/04

Parameter search implemented

## Version 0.3.1

- 2025/07/11

Response plot implemented

## Version 0.4.0

- 2025/07/15

We have rebuilt the code structure with a clear frontend part and backend part to make it easier to understand.

## Version 0.4.1

- 2025/07/16

Loss registration by string complete. YAML read/write fixed.