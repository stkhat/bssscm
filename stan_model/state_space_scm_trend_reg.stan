// TODO:
// - v_*の最後の時点分は不要なのにサンプリングされている状態なので適切に削除 (結果には影響なし)

functions {
  // loss function of covariate matching
  real L_GB(vector beta, real w, real s_y, vector x1, matrix X0, int K
  ) {
    vector[K] error;
    error = x1 - X0' * beta;
    return  -w * (1/(2.0)) * error' * error;
  }
}

data {
  int T_pre;
  int T_prj;
  int T_forth;
  int T;
  int J; // the number of control units
  int K; // the number of covariates
  matrix[J,T] Z;
  vector[T] y;
  matrix[J+1,K] X;
  real<lower=0> w;
  real<lower=0> s_cauchy;

  // Regularized Horseshoe
  real<lower=0> scale_global; // scale for the half - t prior for tau 
  real<lower=1> nu_global; // degrees of freedom for the half - t prior // for tau 
  real<lower=1> nu_local; // degrees of freedom for the half - t priors // for lambdas 
  real<lower=0> slab_scale; // slab scale for the regularized horseshoe 
  real<lower=0> slab_df; // slab degrees of freedom for the regularized // horseshoe 
}

transformed data {
  vector[K] x1;
  matrix[J,K] X0;
  int T_obs;
  // Split covariates for PA and RRD 
   x1 = X[1,]';
   X0 = X[2:(J+1),];

  //  Set the periods of observations
  T_obs = T_pre+T_prj;
}

parameters {
  // simplex[J] beta;
  vector[J] u[T_obs];
  vector[J] v[T_obs];
  real<lower=0> s_u;
  real<lower=0> s_v;
  real<lower=0> s_z;
  real<lower=0> s_y;
  vector[J] z;
  real<lower=0> tau; // global shrinkage parameter 
  vector<lower=0>[J] lambda; // local shrinkage parameter 
  real<lower=0> caux;
}

transformed parameters {
    vector<lower=0>[ J] lambda_tilde; // ’ truncated ’ local shrinkage parameter 
    real<lower=0> c; // slab scale     
    vector[J] beta; // regression coefficients 
    c = slab_scale * sqrt ( caux ); 
    lambda_tilde = sqrt ( c ^2 * square ( lambda ) ./ ( c ^2 + tau ^2* square ( lambda )) ); 
    beta = z .* lambda_tilde * tau; 
}

model {

  // Covariate balancing
  target += L_GB(beta, w, s_y, x1, X0, K);

  // Prior
  // -- half - t priors for lambdas and tau, and inverse - gamma for c ^2
  z ~ normal (0, 1);
  lambda ~ student_t ( nu_local, 0, 1);
  tau ~ student_t ( nu_global, 0, scale_global * s_y );
  caux ~ inv_gamma (0.5* slab_df, 0.5* slab_df );
  s_u ~ cauchy(0, s_cauchy);
  s_z ~ cauchy(0, s_cauchy);
  s_v ~ cauchy(0, s_cauchy);

  // Observation
  // -- Dependent on beta
  for (t in 1:T_pre){
    y[t] ~ normal(u[t]' * beta, s_y);
  }

  // -- Independent on beta
  for (t in 1:T_obs){
    Z[,t] ~ normal(u[t], s_z);
  }

  // -- Transition
  for (t in 2:T_obs){
    v[t] ~ normal(v[t-1], s_v);
    u[t] ~ normal(u[t-1]+v[t-1], s_u);
  }
}

generated quantities {
  vector[J] u_forth[T_forth];
  vector[J] v_forth[T_forth];
  vector[T] y_bsl;

  // Observed
  for (t in 1:T_obs) {
    y_bsl[t] = normal_rng(u[t]' * beta, s_y);
  }

  // Forecast
  // -- first year
  for (j in 1:J){ 
    // Transition
    v_forth[1][j] = normal_rng(v[T_obs][j],s_v);
    u_forth[1][j] = normal_rng(u[T_obs][j]+v[T_obs][j],s_u);
  }
  y_bsl[T_obs+1] = normal_rng(u_forth[1]' * beta, s_y);

  // -- after second year 
  for (t in 2:T_forth) {
    for (j in 1:J){ 
      v_forth[t][j] = normal_rng(v_forth[t-1][j],s_v); 
      u_forth[t][j] = normal_rng(u_forth[t-1][j]+v_forth[t-1][j],s_u);
    }
    y_bsl[T_obs+t] = normal_rng(u_forth[t]' * beta, s_y);
  }
}
