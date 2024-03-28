.libPaths("C:\\Users\\Dan\\Documents\\R_Packages")
library(devtools)
devtools::install_local("E:\\projects\\seqspawnR_with_loadingbars\\SeqSpawnR", force=TRUE)
library(SeqSpawnR)
library(seqinr)


input_file <- "C:\\Users\\Dan\\Desktop\\CDC\\Projects\\dives\\data\\raw\\complete_original_sequences_withsubtypes.fasta"
gen_dir <- "C:\\Users\\Dan\\Desktop\\CDC\\Projects\\dives\\data\\preproc\\generated\\"


N <- 3000 #Number of sequences each subtype should have after generation


# Read sequences from a FASTA file
fasta_sequences <- read.fasta(input_file, whole.header=TRUE, forceDNAtolower = FALSE)

# 1. Count unique subtypes and sequences
subtype_counter <- table(sapply(strsplit(names(fasta_sequences), "\\."), `[`, 1))
print("Subtypes and their counts:")
print(subtype_counter)

# 2. Calculate spawn quotas
spawn_quotas <- (N - subtype_counter) %/% subtype_counter
print("Spawn quotas:")
print(spawn_quotas)

# 3. & 4. Modified loop for spawn quotas and file handling
for (name in names(fasta_sequences)) {
  subtype <- unlist(strsplit(name, "\\."))[1]
  
  output_file <- paste0(gen_dir, subtype, "_generated.fasta")
  
  ref_sequence <- toupper(paste(fasta_sequences[[name]], collapse = ""))
  
  write(paste(">", name, sep = ""), output_file, append = TRUE)
  write(ref_sequence, output_file, append = TRUE)
  
  quota <- spawn_quotas[subtype]
  spawned <- SeqSpawnR::spawn_sequences(quota, seed = ref_sequence, snps = 20)
  
  for (i in 1:length(spawned)) {
    new_name <- paste(name, "_spawned_", i, sep = "")
    spawned_sequence <- toupper(paste(spawned[[i]], collapse = ""))
    
    write(paste(">", new_name, sep = ""), output_file, append = TRUE)
    write(spawned_sequence, output_file, append = TRUE)
  }
}

# 5. Concatenate all output files
output_files <- list.files(path = gen_dir, pattern = "*_generated.fasta", full.names = TRUE)
final_output <- paste0(gen_dir, "final_output.fasta")

for (file in output_files) {
  lines <- readLines(file)
  write(lines, final_output, append = TRUE, ncolumns = 1)
}
