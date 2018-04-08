#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // time when the state is true, in us
  time_us_ = 0.0;

  // state dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_x_;

  //create sigma point matrix
  Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  //create vector for weights
  weights_ = VectorXd(2 * n_aug_ + 1);

}

UKF::~UKF() {}

void UKF::GenerateAugmentedSigmaPoints() {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);


  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }
}

void UKF::SigmaPointPrediction(double delta_t) {

  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

void UKF::PredictMeanAndCovariance() {

  // set weights
  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_+ weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	  /*****************************************************************************
	   *  Initialization
	   ****************************************************************************/
	  if (!is_initialized_) {

	    // first measurement
	    cout << "UKF: " << endl;


	    x_ << 1, 1, 1, 1,0.1;

	    cout << "X matrix init completed" << endl;
	    // init covariance matrix
		P_ << 0.15, 0, 0, 0, 0,
			   0, 0.15, 0, 0, 0,
			   0,    0, 1, 0, 0,
			   0,    0, 0, 1, 0,
			   0, 0, 0, 0, 1;

		cout << "P matrix init completed" << endl;

	    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
	    	double rho = meas_package.raw_measurements_[0];      // range: radial distance from origin
	    	double phi = meas_package.raw_measurements_[1];      // bearing: angle between rho and x axis
	    	double rho_dot = meas_package.raw_measurements_[2];  // radial velocity: change of rho

	    	x_ << rho * cos(phi), rho * sin(phi),0, 0;  // x, y, vx, vy
	    }
	    else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
			/**
			Initialize state.
			*/
	    	double p_x = meas_package.raw_measurements_[0];
	    	double p_y = meas_package.raw_measurements_[1];

	    	x_ << p_x, p_y, 0, 0,0.1;
	    }

	    time_us_ = meas_package.timestamp_;
	    // done initializing, no need to predict or update
	    is_initialized_ = true;

	    cout << "Initialization Finished Successfully :)" << endl;
	    return;
	  }

	  /*****************************************************************************
	  *  Prediction
	  ****************************************************************************/
	  //compute the time elapsed between the current and previous measurements
	  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
	  time_us_ = meas_package.timestamp_;

	  Prediction(dt);

	  /*****************************************************************************
	  *  Update
	  ****************************************************************************/

	  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
		UpdateLidar(meas_package);
	  }
	  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
		UpdateRadar(meas_package);
	  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

	/*****************************************************************************
	*  Augment Sigma Points
	****************************************************************************/
	GenerateAugmentedSigmaPoints();

	/*****************************************************************************
	*  Predict Sigma Points
	****************************************************************************/

	SigmaPointPrediction(delta_t);

	/*****************************************************************************
	*  Convert Predicted Sigma Points to Mean/Covariance
	****************************************************************************/
	PredictMeanAndCovariance();

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	n_z_ = 2;

	LidarMeasurementPrediction();
	double p_x = meas_package.raw_measurements_[0];      // range: radial distance from origin
	double p_y = meas_package.raw_measurements_[1];      // bearing: angle between rho and x axis

	VectorXd z = VectorXd(n_z_);
	z << p_x,p_y;

	Update(z);
}

void UKF::LidarMeasurementPrediction() {

	//create matrix for sigma points in measurement space
	Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);
	//transform sigma points into measurement space
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		//2n+1 simga points
		// extract values for better readibility
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);

		// measurement model
		Zsig(0, i) = p_x;
		Zsig(1, i) = p_y;
	}
	//mean predicted measurement
	z_pred = VectorXd(n_z_);
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}
	//innovation covariance matrix S
	S = MatrixXd(n_z_, n_z_);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		//2n+1 simga points
		//residual
		VectorXd z_diff = Zsig.col(i) - z_pred;
		//angle normalization
		while (z_diff(1) > M_PI)
			z_diff(1) -= 2. * M_PI;
		while (z_diff(1) < -M_PI)
			z_diff(1) += 2. * M_PI;
		S = S + weights_(i) * z_diff * z_diff.transpose();
	}
	//add measurement noise covariance matrix
	MatrixXd R = MatrixXd(n_z_, n_z_);
	R << std_laspx_*std_laspx_, 0, 0, std_laspy_*std_laspy_;
	S = S + R;
}

void UKF::RadarMeasurementPrediction() {

	//create matrix for sigma points in measurement space
	Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);
	//transform sigma points into measurement space
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		//2n+1 simga points
		// extract values for better readibility
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);
		double v1 = cos(yaw) * v;
		double v2 = sin(yaw) * v;
		// measurement model
		Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y); //r
		Zsig(1, i) = atan2(p_y, p_x); //phi
		Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); //r_dot
	}
	//mean predicted measurement
	z_pred = VectorXd(n_z_);
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}
	//innovation covariance matrix S
	S = MatrixXd(n_z_, n_z_);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		//2n+1 simga points
		//residual
		VectorXd z_diff = Zsig.col(i) - z_pred;
		//angle normalization
		while (z_diff(1) > M_PI)
			z_diff(1) -= 2. * M_PI;
		while (z_diff(1) < -M_PI)
			z_diff(1) += 2. * M_PI;
		S = S + weights_(i) * z_diff * z_diff.transpose();
	}
	//add measurement noise covariance matrix
	MatrixXd R = MatrixXd(n_z_, n_z_);
	R << std_radr_ * std_radr_, 0, 0, 0, std_radphi_ * std_radphi_, 0, 0, 0, std_radrd_
			* std_radrd_;
	S = S + R;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	//set measurement dimension, radar can measure r, phi, and r_dot
	n_z_ = 3;

	RadarMeasurementPrediction();
	double rho = meas_package.raw_measurements_[0];      // range: radial distance from origin
	double phi = meas_package.raw_measurements_[1];      // bearing: angle between rho and x axis
	double rho_dot = meas_package.raw_measurements_[2];  // radial velocity: change of rho

	VectorXd z = VectorXd(n_z_);
	z << rho,phi,rho_dot;

	Update(z);
}

void UKF::Update(VectorXd z){

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

	//residual
	VectorXd z_diff = Zsig.col(i) - z_pred;
	//angle normalization
	while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
	while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

	// state difference
	VectorXd x_diff = Xsig_pred_.col(i) - x_;
	//angle normalization
	while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
	while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

	Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();


  //calculate NIS
    double NIS_ = z_diff.transpose() * S.inverse() * z_diff;

    cout << "Radar NIS = " << NIS_ << endl;

}
