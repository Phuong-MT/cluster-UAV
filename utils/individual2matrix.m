function [AssociationMatrix, PowerAllocation] = individual2matrix(Individual, num_sats, num_aps, num_ues)
  % Extract association part and power allocation part from the individual
  assoc_length = (num_sats + num_aps) * num_ues;
  AssociationPart = Individual(1:assoc_length);
  PowerAllocation = Individual(assoc_length + 1:end);

  % Reshape the association part into a matrix
  AssociationMatrix = reshape(AssociationPart, num_sats + num_aps, num_ues)';
end