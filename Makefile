.SUFFIXES:
.SECONDEXPANSION:
.DELETE_ON_ERROR:

THISDIR := $(realpath $(dir $(abspath $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST)))))
CONFIGDIR := $(abspath $(THISDIR)/config)
SCRIPTS := $(THISDIR)/scripts
DATA := $(THISDIR)/data
BUILD := $(THISDIR)/build

ZSPEECH-SAMPLE := $(foreach f, $(shell cat $(DATA)/sample_files.txt | paste -sd ' ' -), build/zerospeech/sample/$(f))
ZSPEECH-ENGLISH := $(foreach f, $(shell cat $(DATA)/english_files.txt | paste -sd ' ' -), build/zerospeech/english/$(f))

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

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

%/user-buckeye-directory.txt: | %
	echo '../buckeye_speech_corpus' > $@

################################################################################
#
# Recipes 
#
################################################################################

zerospeech-%: build/zerospeech/%/mfccs build/zerospeech/%/rasanen_seg.txt scripts/autoencodeDecodeChars.py \
data/%.wrd
	python $(word 3, $^) $(dir $(word 1, $^)) --segfile $(word 2, $^) --goldfile $(word 4, $^) --logfile outputs/$*/ --acoustic --gpufrac None

clean:
	rm -rf build

config build:
	mkdir $@

build/zerospeech/%/:
	mkdir -p $@

outputs/%/:
	mkdir -p $@

build/zerospeech/%/rasanen_seg.txt: scripts/rasanen2seginit.py data/rasanen_seg_%_src.txt
	cat $(word 2, $^) | python $(word 1, $^) > $@

.PRECIOUS: $(ZSPEECH-SAMPLE)
$(ZSPEECH-SAMPLE): $(BUCKEYEDIR) $(CONFIGDIR)/user-buckeye-directory.txt | $$(dir $$@)
	find $(word 1, $^) -name "$(notdir $@)" -type f -exec cp '{}' build/zerospeech/sample/ \;

.PRECIOUS: $(ZSPEECH-ENGLISH)
$(ZSPEECH-ENGLISH): $(BUCKEYEDIR) $(CONFIGDIR)/user-buckeye-directory.txt | $$(dir $$@)
	find $(word 1, $^) -name "$(notdir $@)" -type f -exec cp '{}' build/zerospeech/english/ \;

build/zerospeech/sample/wav-rspecifier.scp: $(ZSPEECH-SAMPLE)
	rm -f $@
	$(foreach wav, $^,echo '$(notdir $(basename $(wav))) $(abspath $(wav))' >> $@;)
	
build/zerospeech/sample/feats-wspecifier.scp: $(ZSPEECH-SAMPLE)
	rm -f $@
	$(foreach wav, $^,echo '$(notdir $(basename $(wav))) $(abspath $(basename $(wav)).mfcc)' >> $@;)
	
build/zerospeech/english/wav-rspecifier.scp: $(ZSPEECH-ENGLISH)
	rm -f $@
	$(foreach wav, $^,echo '$(notdir $(basename $(wav))) $(abspath $(wav))' >> $@;)
	
build/zerospeech/english/feats-wspecifier.scp: $(ZSPEECH-ENGLISH)
	rm -f $@
	$(foreach wav, $^,echo '$(notdir $(basename $(wav))) $(abspath $(basename $(wav)).mfcc)' >> $@;)
	
%mfccs: %wav-rspecifier.scp %feats-wspecifier.scp $(KALDIDIR)/src/featbin/compute-mfcc-feats $(KALDIDIR)/src/featbin/add-deltas $(CONFIGDIR)/user-kaldi-directory.txt
	$(word 3, $^) scp:$(abspath $(word 1, $^)) scp:$(abspath $(word 2, $^))
	$(word 4, $^) scp:$(abspath $(word 2, $^)) scp,t:$(abspath $(word 2, $^))
	echo 'Done' > $@

