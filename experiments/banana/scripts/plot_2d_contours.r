
test_info_grid <- get_init_design_list(inv_prob, design_method="tensor_product_grid",
                                       N_design=50^2)


plt_df <- data.frame(test_info_grid$input, lprior=test_info_grid$lprior,
                     llik=test_info_grid$llik, lpost=test_info_grid$lpost,
                     fwd=test_info_grid$fwd)


plt_fwd <- ggplot(plt_df, aes(x=u1, y=u2, z=fwd)) +
              geom_contour_filled() +
              ggtitle("Forward model")
plt_lprior <- ggplot(plt_df, aes(x=u1, y=u2, z=lprior)) +
                geom_contour_filled() +
                ggtitle("Log Prior Density")
plt_llik <- ggplot(plt_df, aes(x=u1, y=u2, z=llik)) +
              geom_contour_filled() +
              ggtitle("Log Likelihood")
plt_post <- ggplot(plt_df, aes(x=u1, y=u2, z=exp(lpost))) +
              geom_contour_filled() +
              ggtitle("Posterior density")
plt_lpost <- ggplot(plt_df, aes(x=u1, y=u2, z=lpost)) +
              geom_contour_filled() +
              ggtitle("Log Posterior density")


plot(plt_fwd)
plot(plt_lprior)
plot(plt_llik)
plot(plt_post)
plot(plt_lpost)



