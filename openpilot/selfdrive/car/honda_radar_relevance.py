"""Current proximity of an admitted ordinary radar lead."""


HORIZON = 10.0


def loss_relevant(*, distance, ego_speed, path_state):
  """Retain the existing near-lead measurement-expiry protection."""
  near = distance <= max(15.0, 2.0 * ego_speed + ego_speed * ego_speed / 6.0)
  return near and path_state == 'inPath'
