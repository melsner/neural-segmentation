.SUFFIXES:
.SECONDEXPANSION:
.DELETE_ON_ERROR:

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

THISDIR := $(realpath $(dir $(abspath $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST)))))
CONFIGDIR := $(abspath $(THISDIR)/config)
SCRIPTS := $(THISDIR)/scripts
DATA := $(THISDIR)/data

MSG1 := The current config file, 
MSG2 := , points to a non-existent location (
MSG3 := ). Fix before re-running make, or (if not needed) set to ".".

define CONFIGWARN =

ATTENTION! I had to create $(CONFIG),
which likely contains an incorrect default value.
Targets with dependencies to this file will fail until it is fixed.

endef

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-kaldi-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
KALDIDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(KALDIDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(KALDIDIR)$(MSG3))
endif
endif

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-lexicondiscovery-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
LEXICONDISCOVERYDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(LEXICONDISCOVERYDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(LEXICONDISCOVERYDIR)$(MSG3))
endif
endif

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-rasanen-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
RASANENDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(RASANENDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(RASANENDIR)$(MSG3))
endif
endif

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-buckeye-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
BUCKEYEDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(BUCKEYEDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(BUCKEYEDIR)$(MSG3))
endif
endif



%/user-kaldi-directory.txt: | %   
	echo '../kaldi' > $@

%/user-lexicondiscovery-directory.txt: | %
	echo '../lexicon_discovery' > $@

%/user-rasanen-directory.txt: | %
	echo '../rasanen' > $@

%/user-buckeye-directory.txt: | %
	echo '../buckeye_speech_corpus' > $@

################################################################################
#
# Recipes 
#
################################################################################

config data/zerospeech:
	mkdir $@

data/zerospeech/%/: | data/zerospeech
	mkdir $@

data/zerospeech/sample/get-wavs: $(SCRIPTS)/get_wavs_from_list.sh $(DATA)/sample_files.txt $(BUCKEYEDIR) $(CONFIGDIR)/user-buckeye-directory.txt | $$(dir $$@)
	$(word 1, $^) $(word 2, $^) $(word 3, $^) $(dir $@)

data/zerospeech/english/get-wavs: $(SCRIPTS)/get_wavs_from_list.sh $(DATA)/english_files.txt $(BUCKEYEDIR) $(CONFIGDIR)/user-buckeye-directory.txt | $$(dir $$@)
	$(word 1, $^) $(word 2, $^) $(word 3, $^) $(dir $@)

/%wav-rspecifier.scp: $$(wildcard /%*.wav)
	rm -f $@
	$(foreach wav, $^,echo '$(notdir $(basename $(wav))) $(wav)' >> $@;)

/%feats-wspecifier.scp: $$(wildcard /%*.wav)
	rm -f $@
	$(foreach wav, $^,echo '$(notdir $(basename $(wav))) $(basename $(wav)).mfcc' >> $@;)

/%mfccify: /%wav-rspecifier.scp /%feats-wspecifier.scp $(KALDIDIR)/src/featbin/compute-mfcc-feats $(KALDIDIR)/src/featbin/add-deltas $(CONFIGDIR)/user-kaldi-directory.txt
	$(word 3, $^) scp:$(word 1, $^) scp:$(word 2, $^)
	$(word 4, $^) scp:$(word 2, $^) scp,t:$(word 2, $^)

%mfccify: $$(abspath $$@);
