/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author of framework: Tiffany Huang
 * Last modified: Ying Li
 */
#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::unordered_map;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  particles.resize(num_particles);
  
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x{x, std[0]};
  std::normal_distribution<double> dist_y{y, std[1]};
  std::normal_distribution<double> dist_theta{theta, std[2]};
  
  // Generates initial particles around GPS position with random Gaussian noise.
  std::generate_n(particles.begin(), num_particles, [&](){
    Particle p;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;
    return p;
  });
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Predict the state of next timestamp for each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x{0, std_pos[0]};
  std::normal_distribution<double> dist_y{0, std_pos[1]};
  std::normal_distribution<double> dist_theta{0, std_pos[2]};
  
  for (auto& particle : particles) {
    double theta1 = particle.theta;
    double theta2 = theta1 + yaw_rate * delta_t;
      
    if (std::fabs(yaw_rate) < 0.0001) {
      // Overcomes divide by zero error: when yaw_rate is too small, treat it as
      // a constant yaw motion model
      particle.x += velocity * delta_t * std::cos(theta1);
      particle.y += velocity * delta_t * std::sin(theta1);
    } else {
      // Normal case of motion model
      particle.x += (velocity / yaw_rate) * (std::sin(theta2) - std::sin(theta1));
      particle.y += (velocity / yaw_rate) * (std::cos(theta1) - std::cos(theta2));
    }
    
    // Adds uncertainty noise
    particle.x += dist_x(gen);
    particle.y += dist_y(gen);
    particle.theta = theta2 + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations,
                                     unordered_map<int, std::pair<double, double>>& obs_diff) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (auto& obs : observations) {
    double min_dist = std::numeric_limits<double>::max();
    double diff_x = std::numeric_limits<double>::max();
    double diff_y = std::numeric_limits<double>::max();
    int min_id = -1;
    for (const auto& pred : predicted) {
      double distance = dist(obs.x, obs.y, pred.x, pred.y);
      if (distance < min_dist) {
        min_dist = distance;
        diff_x = obs.x - pred.x;
        diff_y = obs.y - pred.y;
        min_id = pred.id;
      }
    }
    obs.id = min_id;
    obs_diff[min_id] = std::make_pair(diff_x, diff_y);
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double weight_sum{0.0};
  
  for (auto& particle : particles) {
    
    // Transforms from vehicle coordinate system to world coordinate system.
    vector<LandmarkObs> obsns_wcs;
    std::transform(observations.begin(), observations.end(),
                   std::back_inserter(obsns_wcs),
                   [&](const LandmarkObs& obs){
                     double x = std::cos(particle.theta) * obs.x - std::sin(particle.theta) * obs.y + particle.x;
                     double y = std::sin(particle.theta) * obs.x + std::cos(particle.theta) * obs.y + particle.y;
                     return LandmarkObs{0, x, y};
                   });
    
    // Transforms from single_landmark_s to LandmarkObs.
    vector<LandmarkObs> predicted_lms;
    std::transform(map_landmarks.landmark_list.begin(), map_landmarks.landmark_list.end(),
                  std::back_inserter(predicted_lms),
                  [](const Map::single_landmark_s& slm){
                    return LandmarkObs{slm.id_i, slm.x_f, slm.y_f};
                  });

    // Associates landmark data to observation data
    unordered_map<int, std::pair<double, double>> obs_diff;
    dataAssociation(predicted_lms, obsns_wcs, obs_diff);
    
    // Keeps down association results in particle
    vector<int> associations;
    std::transform(obsns_wcs.begin(), obsns_wcs.end(), std::back_inserter(associations),
                   [](const LandmarkObs& obs){ return obs.id; });
    vector<double> sense_x;
    std::transform(obsns_wcs.begin(), obsns_wcs.end(), std::back_inserter(sense_x),
                   [](const LandmarkObs& obs){ return obs.x; });
    vector<double> sense_y;
    std::transform(obsns_wcs.begin(), obsns_wcs.end(), std::back_inserter(sense_y),
                   [](const LandmarkObs& obs){ return obs.y; });
    
    SetAssociations(particle, associations, sense_x, sense_y);
    
    // Calculates particle weight based on multivariant probability
    particle.weight = 1.0;
    for (const auto& obs : obsns_wcs) {
      auto xy_diff = obs_diff[obs.id];
      double multiprob = multiv_prob(0.0, 0.0, xy_diff.first, xy_diff.second, std_landmark[0], std_landmark[1]);
      particle.weight *= multiprob;
    }
    weight_sum += particle.weight;
  }
  
  // Normalizes particle weights
  for (auto& particle : particles) {
    particle.weight /= weight_sum; 
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<double> weights;
  std::transform(particles.begin(), particles.end(),
                std::back_inserter(weights),
                [](const Particle& particle){ return particle.weight; }); 
  
  std::default_random_engine gen;
  std::discrete_distribution<> d(weights.begin(), weights.end());
  vector<Particle> new_particles;
  new_particles.resize(num_particles);
  std::generate_n(new_particles.begin(), new_particles.size(), [&](){
    int idx = d(gen);
    return particles[idx];
  });
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}