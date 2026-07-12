# Response to Reviewers

**Manuscript:** Geo-ORBIT / FedMeta-GeoLane (IEEE T-ITS ms 25-06-2863)

We thank the Associate Editor and the reviewer for a careful reading and for comments that identified real limitations in the submitted version. We have treated each comment as a result to produce rather than a point to argue, and the revision adds new experiments, an annotation-based evaluation against human labels, and several implementation corrections that we describe below. Unless noted otherwise, every quantitative result in this letter is reported as a mean over three training seeds, all geometric quantities are expressed in meters, and all reported quantities are independent of any model's learned weighting. Reviewer comments are paraphrased in italics at the head of each item, followed by our response and pointers to the revised manuscript.

---

## R1.1 Lane count error (marked critical)

> *The lane count error remains unresolved and is a critical shortcoming of the method.*

In preparing this revision we re-examined the metrics reported in the submitted Table 1 and found that the loss components were scaled by each model's learned weighting parameters, while the baseline used unnormalized weights, so the reported totals were not directly comparable across models. Part of the apparent margin reflected the weighting convention rather than a difference in detected geometry. We have rebuilt the evaluation so that every reported quantity is model-independent and expressed in meters, and we report each result as a mean with standard deviation over three training seeds. Under the original weighted protocol the corrected implementation still ranks the federated model lowest, roughly 77 percent below the baseline total, so the comparison reported in the submission is preserved in direction. Under the corrected protocol the detected geometry is comparable across the baseline, the per-camera meta model, and the federated meta model, all falling within about 3 m of independent human annotations, which we show is the precision ceiling set by the reference map rather than by the detection method. Against those annotations the adaptive models reach a lane-level F1 of 0.76 on the training sites against 0.70 for the fixed baseline, with the gain coming from recall of real lanes. The corrected magnitudes are smaller than the originally reported weighted totals, and we describe the correction and the model-independent protocol in the revised manuscript [Section X, Table N]. Because the geometry is reference-floored, the methods are distinguished instead by their lane count accuracy, by deployment stability, and by communication cost, which the remaining points and Section X address.

We addressed the lane count error directly at its source. The peak prominence that governs how many lane clusters the histogram yields is now a learned, scene-adaptive parameter rather than a fixed constant. The detection histogram is normalized by its own maximum before peak finding, so the learned prominence is a scale-free fraction that transfers across sparse and busy scenes. This targets the under- and over-segmentation the reviewer identified, and the revised method text and equation describe the normalized-prominence formulation the implementation uses [Section X, Eq. N].

We report a dedicated lane count analysis rather than folding the count into a weighted loss. Table N gives per-site exact-match accuracy and the mean absolute lane count error, reported separately for seen and unseen sites for all three models over three seeds. On the seen sites both adaptive models reduce the error to close to one lane, 1.06 with standard deviation 0.01 for the per-camera meta model and 1.12 with 0.02 for the federated model, against 2.33 for the fixed baseline. On the unseen sites the two adaptive models are comparable, 4.50 with 0.54 and 4.83 with 0.12 respectively, and the unseen counts are inflated by the reference networks at those sites rather than by the detector, consistent with the decomposition under R1.3. A residual gap therefore remains at sparse or unbalanced sites, which we quantify under R1.2.

On fusing the OSM lane count as a prior on k, we considered this and did not adopt it, for a reason our own analysis supports. Our reference audit against human annotations shows that the OSM reference is itself unreliable at several sites, with offsets between 2.6 and 65.9 m and a cross-site mean near 26 m, and with lane counts inflated at some locations, so binding k to the OSM tag would import that error into the detection. We instead rely on trajectory evidence with the adaptive prominence above, which keeps the method label-free and does not inherit reference count errors. We can add an optional soft prior if the reviewer prefers, and we note the tradeoff in the text [Section X].

---

## R1.2 Sparse and unbalanced lanes

> *Sparse or unbalanced lanes, such as the leftmost lane at the Monona site, are missed by the method.*

We separate two effects the reviewer raises, detection precision and detection recall. Using human annotations as an independent reference, the detected lane geometry is accurate to a cross-site mean of 2.9 m, so the precision of what the method detects is not the limitation. The limitation is recall. At a 5 m match threshold the adaptive models recover 68 percent of the annotated lanes at the training sites against 60 percent for the fixed baseline, and about 53 percent at held-out sites, and a lane-by-lane analysis attributes the remaining misses primarily to lanes with little or no trajectory evidence in the observation period, which is exactly the Monona case. The adaptive prominence above lowers the recovery threshold for weak lanes, and our sensitivity analysis shows that the weakest lanes are limited by trajectory sampling rather than by any detection parameter, since they remain unrecovered at the most aggressive settings.

To bound this quantitatively we vary the observed traffic volume directly, retraining with a seeded fraction of each site's vehicle trajectories over three seeds. Detection F1 against the annotations remains within 0.03 of the full-data value at every fraction down to one tenth of the recorded traffic, with a decrease of 0.018 at one tenth on the training sites, so reliable detection persists far below the recorded demand and the binding constraint is lanes that receive no traffic at all within the observation period, which longer accumulation addresses [Section X, Fig. N]. This dependence is inherent to any trajectory-based method, including [Ren, 29] and Qiu et al. [28], and we now bound it rather than leave it qualitative.

---

## R1.3 Homography and calibration sensitivity

> *The approach is sensitive to the quality of the homography and camera calibration.*

We turned the calibration concern into a measured analysis. First we corrected a defect in the homography fitting in which the robust estimator threshold was expressed in target units and therefore never rejected any control point. With the corrected estimator, a deterministic robust fit, and a geometric sanity gate, the calibration is stable across runs and rejects mis-picked control points. We report the number of ground control points per site and a leave-one-out reprojection audit that flags the specific points that degrade each site [Section X, Table N].

Second, and more importantly, we use human lane annotations to decompose each site's apparent geometric error into a detection component and a reference component. Detection error, measured with annotations and detections passed through the same homography so that calibration cancels, has a cross-site mean of 2.9 m over the twelve annotated sites, while the offset between the OSM reference and the annotations ranges from 2.6 to 65.9 m with a cross-site mean near 26 m. This shows that the residual site error is dominated by the schematic OSM reference rather than by calibration or detection, and it gives a measured tolerance rather than a qualitative worry.

The same audit yields a uniform and model-independent site admission rule for the evaluation. A site enters the train and test split when its annotation-based detection error under the fixed baseline configuration is below 5 m. Two sites fail the rule, at 6.8 and 7.1 m, both attributable to ground control points picked near the horizon where the homography is least constrained, and we retain them as calibration case studies rather than in the comparison tables. Every admitted site measures between 0.2 and 4.3 m.

Taken together, the leave-one-out reprojection audit and this detection-versus-reference decomposition give a direct measurement of how much calibration contributes to each site's error, which is the calibration sensitivity the reviewer asked us to characterize. The decomposition localizes the residual error to the schematic reference at the two excluded sites and confirms that at admitted sites calibration is not the binding source of error [Section X].

---

## R1.4 Static digital twin after OSM import

> *The digital twin is static after the initial OSM import and does not update with observed conditions.*

We implemented and demonstrate one closed-loop dynamic update. At the Monona site we synthesize a lane closure in the input data by removing the vehicles of one detected lane, and we disclose this construction explicitly since no physical closure occurred during the recording. The demonstrated closure is a synthesized scenario, while the loop that responds to it is exercised in full. The change detector compares the detected lane sets of consecutive observation windows and flags the disappeared lane, the flag carries the lane's mapping into the SUMO network, the closure is applied to the running simulation through TraCI without any re-import of the network, and the scene is re-simulated under identical demand and seed. Under identical demand and seed the closure raises the mean departure delay from 34.1 to 36.2 seconds and reduces the vehicles served within the thirty-minute horizon from 2964 to 2946, while the delay inside the network is absorbed by the remaining lanes. The twin thereby quantifies the operational consequence of the detected change, in this case a modest and entry-concentrated penalty at the simulated demand level. The mapping from detected lanes to network lanes uses travel direction and lateral rank, which are invariant to the schematic offset of the OSM reference that we quantify under R1.3. The demand is the osmWebWizard trip set scaled by a factor of sixty, which we state in the text. Broader dynamic updates remain future work, and we have softened the high-fidelity language in the manuscript accordingly [Section X].

---

## R1.5 Additional references

> *The related work should include recent work on federated-learning-based digital twins.*

We have added the two suggested references on federated-learning-based digital twins and positioned them within the related work, [REF: Y. Gong et al., IEEE TWC 2025, doi 10.1109/TWC.2025.3548574] and [REF: Y. Gong et al., IEEE TMC 2025, doi 10.1109/TMC.2024.3521399]. Please confirm these are the intended works [Section X].

---

## Other proactive corrections

### Reconciliation of implementation and manuscript

As part of the revision we carried out a verification pass that reconciled the implementation with the manuscript and corrected four items. The reported metrics are now model-independent and in meters rather than scaled by learned weights, as described under R1.1. The contrastive centerline term now uses a genuine hard negative, the nearest reference lane other than the matched one, and the corresponding equation text is revised to describe the distance-based formulation the method actually uses rather than a learned embedding the implementation does not contain. The reference lane width is read from the road network rather than a fixed constant. The homography estimator threshold is expressed in pixel units so that robust fitting rejects mis-picked control points, as described under R1.3. Each change is reflected in the revised equations and text [Section X, Eqs. N].

### External baseline

We added a comparison against the multiple-ROI lane learning system of Qiu et al. [28], using the authors' public implementation and evaluating it under the same annotation-based protocol as our methods. To make the comparison detector-controlled we run their lane learning on exactly the detections our system consumes, so the difference measures lane learning rather than detection quality. Their method reaches a lane-level F1 of 0.52 on our training sites and 0.22 on the held-out sites at the 5 m threshold, against 0.70 for our fixed baseline and 0.76 for the adaptive models on the training sites and 0.63 at held-out sites. The per-site behavior is informative. On clean highway views inside its design assumptions the method is competitive, reaching 0.80 at one held-out interstate site, while it returns no lanes at two arterial sites whose geometry violates its highway ROI assumptions and places lanes with a large systematic offset at one further site. We selected this method because it is the closest published system to our setting and has a public authors' implementation that permits a faithful comparison.

For [Ren, 29], which has no public implementation, we implemented the comparable lane extraction core, incremental clustering of tracked trajectories under Hausdorff distance, and evaluated it on the same detections with self-calibrated thresholds. This core is a strong baseline on raw centerline recall, reaching 0.72 on the training sites and 0.68 at held-out sites, at uniformly lower precision than our methods and without lane widths, boundaries, or the map anchoring that the digital twin integration requires. Its self-calibration constants were fixed before evaluation, and a sensitivity check over a plus or minus twenty percent range of the clustering threshold and a two to five percent membership floor keeps its F1 between 0.64 and 0.78 on the training sites and between 0.64 and 0.72 at held-out sites, so the reported row neither understates nor overstates the method. Its strength at sites with merging lane geometry is itself informative. It indicates that the remaining recall gap of our pipeline lies in the histogram-based separation stage rather than in the trajectory evidence, and it identifies full-trajectory clustering as a natural successor design for the detection stage, which we discuss as future work [Section X].

We also report the OSM reference itself as a comparison row under the same protocol, where it reaches 0.63 on the training sites but only 0.31 at held-out sites, which quantifies the staleness of the schematic map that all of the behavior-based methods, including the baselines, substantially exceed at new locations [Table N].

---

We thank the reviewer again for comments that materially improved the rigor of the evaluation and the honesty of the reported results. We are happy to provide any further clarification or additional experiments the reviewer considers useful.
