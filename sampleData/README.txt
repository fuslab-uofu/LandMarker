Register moving image to fixed image using rigid, landmarks-based registration
------------------------------------------------------------------------------
1. Load moving image 
		Moving Volume > Load NIfTI > moving.nii.gz
2. Load fixed image 
		Fixed Volume > Load NIfTI > fixed.nii.gz
3. Select rigid registration type 
		Registration > Landmark-Based Algorithms > Rigid
4. Load rigid registration landmarks (moving first, then fixed) 
		Landmarks > Import Individual json Files > 	rigidLandmarks\moving_lm_for_fft.json
													rigidLandmarks\fixed_lm_for_rigid.json
5. Register 
6. Save registered image 
		Registered Volume > Save NIfTI
		
Further register moving image to fixed image using FFT rigid-them-splines registration
--------------------------------------------------------------------------------------
7. Load registered image as a new moving image. Fixed image stays the same.
8. Select FFT splines registration
		Registration > Landmark-Based Algorithms > FFT Projection > Rigid, then Splines
9. Load FFT registration landmarks
		Landmarks > Import Individual json Files > 	FFTRigidSplinesLandmarks\moving_lm_for_fft.json
													FFTRigidSplinesLandmarks\fixed_lm_for_fft.json
10. Register
11. Save registered image



